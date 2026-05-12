"""
Pinecone Medical Knowledge Base
================================
Manages the PCES organisation's Pinecone vector index.

- One index  : pces-medical-kb  (Azure serverless, dim=1024, cosine)
- One namespace per department (neurology, general_medicine, cardiology,
  dentist, pulmonology)
- Embedding  : text-embedding-3-small with dimensions=1024 (OpenAI matryoshka
  truncation — matches the pre-created index dimension exactly)

Usage
-----
from pinecone_kb import get_pinecone_kb, PINECONE_KB_AVAILABLE

if PINECONE_KB_AVAILABLE:
    kb = get_pinecone_kb()
    results = kb.query("symptoms of atrial fibrillation", namespace="cardiology")
"""

from __future__ import annotations

import os
import time
import uuid
import logging
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Availability guard — everything is wrapped so the app starts without Pinecone
# ---------------------------------------------------------------------------
try:
    from pinecone import Pinecone, ServerlessSpec
    _PINECONE_IMPORTABLE = True
except ImportError:
    _PINECONE_IMPORTABLE = False
    logger.warning("pinecone package not installed — Pinecone KB unavailable")

try:
    from openai import OpenAI as _OpenAIClient
    _OPENAI_IMPORTABLE = True
except ImportError:
    _OPENAI_IMPORTABLE = False

PINECONE_KB_AVAILABLE: bool = _PINECONE_IMPORTABLE and _OPENAI_IMPORTABLE

# ---------------------------------------------------------------------------
# Constants (read from env)
# ---------------------------------------------------------------------------
_API_KEY     = os.getenv("PINECONE_API_KEY", "")
_REGION      = os.getenv("PINECONE_REGION", "eastus2")
_CLOUD       = os.getenv("PINECONE_ENVIRONMENT", "azure")
_INDEX_NAME  = os.getenv("PINECONE_INDEX_NAME", "pces-medical-kb")
_DIMENSION   = int(os.getenv("PINECONE_INDEX_DIMENSION", "1024"))
_METRIC      = os.getenv("PINECONE_METRIC", "cosine")
_ORG         = os.getenv("ORGANIZATION_NAME", "PCES")
_DEPARTMENTS = [
    d.strip()
    for d in os.getenv("PINECONE_DEPARTMENTS",
                        "neurology,general_medicine,cardiology,dentist,pulmonology").split(",")
    if d.strip()
]

# Embedding model that natively supports 1024-dim output via truncation
_EMBED_MODEL = "text-embedding-3-small"
_EMBED_DIM   = _DIMENSION  # 1024


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """Naive character-level chunker."""
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class PineconeMedicalKB:
    """
    PCES Pinecone knowledge base client.

    Thread-safe after __init__. All public methods return plain Python objects
    so callers never need to import pinecone directly.
    """

    def __init__(self) -> None:
        if not PINECONE_KB_AVAILABLE:
            raise RuntimeError("Pinecone or OpenAI package not available")
        if not _API_KEY:
            raise RuntimeError("PINECONE_API_KEY not set in environment")

        self._pc      = Pinecone(api_key=_API_KEY)
        self._oai     = _OpenAIClient(api_key=os.getenv("openai_api_key", ""))
        self._index   = self._ensure_index()
        self.departments: List[str] = _DEPARTMENTS

    # ------------------------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------------------------

    def _ensure_index(self):
        """Create the Pinecone index if it doesn't exist, then return it."""
        existing = [idx.name for idx in self._pc.list_indexes()]
        if _INDEX_NAME not in existing:
            logger.info("Creating Pinecone index '%s' …", _INDEX_NAME)
            self._pc.create_index(
                name=_INDEX_NAME,
                dimension=_DIMENSION,
                metric=_METRIC,
                spec=ServerlessSpec(cloud=_CLOUD, region=_REGION),
            )
            # Wait until ready
            for _ in range(30):
                desc = self._pc.describe_index(_INDEX_NAME)
                if desc.status.get("ready", False):
                    break
                time.sleep(2)
            logger.info("Pinecone index '%s' is ready.", _INDEX_NAME)
        else:
            logger.info("Pinecone index '%s' already exists.", _INDEX_NAME)

        return self._pc.Index(_INDEX_NAME)

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts using text-embedding-3-small at 1024 dims."""
        if not texts:
            return []
        response = self._oai.embeddings.create(
            model=_EMBED_MODEL,
            input=texts,
            dimensions=_EMBED_DIM,
        )
        return [item.embedding for item in response.data]

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def upsert_documents(
        self,
        texts: List[str],
        namespace: str,
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 100,
    ) -> int:
        """
        Embed and upsert plain text documents into a namespace.

        Args:
            texts:         List of text strings to index.
            namespace:     Pinecone namespace (department name).
            metadata_list: Optional per-document metadata dicts.
            batch_size:    Vectors per upsert batch.

        Returns:
            Number of vectors upserted.
        """
        if not texts:
            return 0

        metadata_list = metadata_list or [{} for _ in texts]
        total = 0

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i: i + batch_size]
            batch_meta  = metadata_list[i: i + batch_size]

            embeddings = self.embed(batch_texts)
            vectors = [
                {
                    "id":       str(uuid.uuid4()),
                    "values":   emb,
                    "metadata": {**meta, "text": txt[:500], "namespace": namespace},
                }
                for txt, emb, meta in zip(batch_texts, embeddings, batch_meta)
            ]
            self._index.upsert(vectors=vectors, namespace=namespace)
            total += len(vectors)

        logger.info("Upserted %d vectors into namespace '%s'", total, namespace)
        return total

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        namespace: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Query the Pinecone index.

        Args:
            query_text: Natural-language query.
            namespace:  If given, search only that namespace.
                        If None, search all configured department namespaces
                        and merge results.
            top_k:      Max results per namespace.

        Returns:
            List of dicts with keys: text, score, namespace, metadata.
        """
        query_emb = self.embed([query_text])[0]
        namespaces = [namespace] if namespace else _DEPARTMENTS
        results: List[Dict[str, Any]] = []

        for ns in namespaces:
            try:
                resp = self._index.query(
                    vector=query_emb,
                    top_k=top_k,
                    namespace=ns,
                    include_metadata=True,
                )
                for match in resp.get("matches", []):
                    results.append({
                        "text":      match.get("metadata", {}).get("text", ""),
                        "score":     match.get("score", 0.0),
                        "namespace": ns,
                        "metadata":  match.get("metadata", {}),
                    })
            except Exception as exc:
                logger.warning("Pinecone query failed for namespace '%s': %s", ns, exc)

        # Sort merged results by score descending, return top_k overall
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return index statistics (vector counts per namespace)."""
        try:
            raw = self._index.describe_index_stats()
            return {
                "index":       _INDEX_NAME,
                "dimension":   _DIMENSION,
                "total_vectors": raw.get("total_vector_count", 0),
                "namespaces":  {
                    ns: info.get("vector_count", 0)
                    for ns, info in raw.get("namespaces", {}).items()
                },
            }
        except Exception as exc:
            logger.error("Pinecone stats error: %s", exc)
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Sample-data seeder
    # ------------------------------------------------------------------

    def seed_sample_data(self, force_reseed: bool = False) -> Dict[str, int]:
        """
        Populate each department namespace with a handful of sample medical
        documents so that test queries return real results.
        Only seeds if the namespace is currently empty (unless force_reseed=True).
        """
        sample_data: Dict[str, List[str]] = {
            "cardiology": [
                "Atrial fibrillation (AF) is the most common sustained cardiac arrhythmia, "
                "characterised by irregular and often rapid heart rate. Management includes "
                "rate control, rhythm control, and anticoagulation to prevent stroke.",
                "Supraventricular tachycardia (SVT) is a rapid heart rhythm arising above "
                "the bundle of His. Acute termination: vagal manoeuvres are first-line. "
                "Adenosine 6 mg IV is the drug of choice for acute SVT. Diltiazem and "
                "verapamil (non-dihydropyridine calcium channel blockers) are accepted as "
                "treatments of choice for termination of SVT when adenosine fails or is "
                "contraindicated. Beta-blockers are an alternative. DC cardioversion is "
                "used for haemodynamically unstable patients.",
                "Heart failure is a clinical syndrome in which the heart is unable to pump "
                "sufficient blood to meet the body's needs. Treatment includes ACE inhibitors, "
                "beta-blockers, and diuretics.",
                "Hypertension guidelines recommend lifestyle modifications as first-line therapy "
                "including sodium restriction, regular aerobic exercise, and weight management. "
                "Pharmacotherapy starts with ACE inhibitors, ARBs, or thiazide diuretics.",
                "Coronary artery disease (CAD) results from atherosclerotic plaque in coronary "
                "arteries. Diagnosis involves ECG, stress testing, and coronary angiography. "
                "Treatment includes statins, antiplatelet therapy, and revascularisation.",
                "Cardiac rehabilitation is a medically supervised program for patients recovering "
                "from heart attack, heart failure, or cardiac surgery. It includes exercise, "
                "education, and psychological support.",
            ],
            "neurology": [
                "Ischemic stroke occurs when blood supply to part of the brain is cut off. "
                "IV alteplase (tPA) within 4.5 hours of symptom onset is standard thrombolytic "
                "therapy. Mechanical thrombectomy is indicated for large vessel occlusion.",
                "Migraine is a primary headache disorder characterised by recurrent unilateral "
                "throbbing pain with nausea, photophobia, and phonophobia. Triptans are the "
                "mainstay of acute treatment; topiramate and propranolol are used prophylactically.",
                "Epilepsy is a neurological disorder marked by recurrent unprovoked seizures. "
                "Antiseizure medications (ASMs) such as levetiracetam, lamotrigine, and valproate "
                "are first-line agents. Drug-resistant epilepsy may benefit from surgery.",
                "Parkinson's disease is a progressive neurodegenerative disorder characterised by "
                "tremor, rigidity, bradykinesia, and postural instability. Dopaminergic therapy "
                "with levodopa/carbidopa remains the cornerstone of treatment.",
                "Multiple sclerosis (MS) is an autoimmune demyelinating disease of the CNS. "
                "Disease-modifying therapies (DMTs) such as interferon-beta, glatiramer acetate, "
                "and natalizumab reduce relapse rates and delay disability progression.",
            ],
            "general_medicine": [
                "Type 2 diabetes mellitus symptoms include polyuria (frequent urination), "
                "polydipsia (excessive thirst), polyphagia (increased hunger), unexplained "
                "weight loss, fatigue, blurred vision, slow-healing wounds, and recurrent "
                "infections. Many patients are asymptomatic at diagnosis. Risk factors include "
                "obesity, family history, sedentary lifestyle, and age over 45.",
                "Type 2 diabetes mellitus management includes lifestyle modification, metformin "
                "as first-line pharmacotherapy, and addition of GLP-1 agonists or SGLT-2 "
                "inhibitors for cardiovascular benefit in high-risk patients.",
                "Pneumonia is an infection of the lung parenchyma. Community-acquired pneumonia "
                "in non-severe cases is treated with amoxicillin or a macrolide. CURB-65 score "
                "guides hospitalisation decisions.",
                "Anaemia is defined as haemoglobin <13 g/dL in men and <12 g/dL in women. "
                "Iron deficiency anaemia is the most common type and is treated with oral "
                "ferrous sulphate. B12 deficiency causes macrocytic anaemia.",
                "Chronic kidney disease (CKD) staging uses GFR and albuminuria. Management "
                "includes blood pressure control, ACE inhibitors, dietary protein restriction, "
                "and monitoring for complications such as anaemia and bone disease.",
                "Hypothyroidism presents with fatigue, weight gain, cold intolerance, and "
                "constipation. Diagnosis is confirmed by elevated TSH. Treatment is with "
                "levothyroxine sodium titrated to normalise TSH.",
            ],
            "dentist": [
                "Dental caries (tooth decay) results from acid produced by oral bacteria. "
                "Prevention includes fluoride toothpaste, sealants, and dietary counselling. "
                "Treatment ranges from fluoride application to fillings, crown, or extraction.",
                "Periodontal disease ranges from gingivitis (reversible gum inflammation) to "
                "periodontitis (irreversible bone loss). Scaling and root planing is the "
                "primary non-surgical treatment for moderate-to-severe periodontitis.",
                "Dental abscess is a localised collection of pus caused by bacterial infection. "
                "Treatment involves drainage (incision or root canal), antibiotics in cases "
                "of spreading infection, and analgesia.",
                "Temporomandibular joint (TMJ) disorders present with jaw pain, clicking, and "
                "limited mouth opening. Conservative management includes occlusal splints, "
                "physiotherapy, and NSAIDs.",
                "Oral cancer risk factors include tobacco, alcohol, and HPV infection. "
                "Early detection via regular screening improves prognosis. Treatment involves "
                "surgery, radiotherapy, and chemotherapy depending on stage.",
            ],
            "pulmonology": [
                "Asthma is a chronic inflammatory airway disease characterised by reversible "
                "airflow obstruction. Step-wise management uses inhaled corticosteroids (ICS) "
                "as controller therapy and short-acting beta-2 agonists (SABA) for rescue.",
                "COPD (chronic obstructive pulmonary disease) is progressive and largely "
                "irreversible. Long-acting bronchodilators (LABAs, LAMAs) are first-line "
                "maintenance therapy. Pulmonary rehabilitation improves exercise capacity.",
                "Pulmonary embolism (PE) presents with dyspnoea, pleuritic chest pain, and "
                "hypoxia. CTPA is the diagnostic standard. Treatment is anticoagulation; "
                "massive PE with haemodynamic compromise warrants thrombolysis.",
                "Obstructive sleep apnoea (OSA) is characterised by repetitive upper airway "
                "collapse during sleep. CPAP therapy is the gold standard treatment. "
                "Weight loss significantly reduces severity in obese patients.",
                "Interstitial lung disease (ILD) encompasses a heterogeneous group of disorders "
                "characterised by fibrosis or inflammation of the lung parenchyma. "
                "Nintedanib and pirfenidone slow progression of idiopathic pulmonary fibrosis.",
            ],
        }

        stats_before = self.stats().get("namespaces", {})
        seeded: Dict[str, int] = {}

        for dept, docs in sample_data.items():
            existing_count = stats_before.get(dept, 0)
            if existing_count > 0 and not force_reseed:
                logger.info("Namespace '%s' already has %d vectors — skipping seed.", dept, existing_count)
                seeded[dept] = 0
                continue

            if existing_count > 0 and force_reseed:
                logger.info("Namespace '%s' has %d vectors — force re-seeding.", dept, existing_count)

            meta = [{"department": dept, "source": f"PCES_{dept}_sample", "source_type": "pinecone"}
                    for _ in docs]
            count = self.upsert_documents(docs, namespace=dept, metadata_list=meta)
            seeded[dept] = count

        return seeded


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_kb_instance: Optional[PineconeMedicalKB] = None


def get_pinecone_kb() -> PineconeMedicalKB:
    """Return (and lazily initialise) the shared PineconeMedicalKB instance."""
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = PineconeMedicalKB()
    return _kb_instance
