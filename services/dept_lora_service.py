"""
Department LoRA Service
=======================
Loads and caches department-specific LoRA fine-tuned models (microsoft/phi-2 + PEFT adapters
produced by the SFT pipeline).  Intended to be used as a Flask app-level singleton stored at
``app.config["DEPT_LORA_SERVICE"]``.

Usage::

    from services.dept_lora_service import DeptLoRAService

    dept_lora_service = DeptLoRAService()                    # at startup
    app.config["DEPT_LORA_SERVICE"] = dept_lora_service

    # at query time
    if dept_lora_service.is_available("Cardiology"):
        result = dept_lora_service.generate("Cardiology", query)
        response_text = result["response"]

Design notes
------------
* LRU cache capped at MAX_CACHED_MODELS (default 3) — each phi-2 LoRA is ~300 MB on CPU.
  When the cache is full, the least-recently-used dept model is evicted and freed.
* Model loading is guarded by a threading.Event so concurrent requests for the same dept
  don't trigger duplicate loads.
* A per-load threading timeout (default 120 s) prevents a slow download from hanging requests.
* All failures are caught and logged; the caller gets ``{"success": False, ...}``.
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from typing import Any

from utils.error_handlers import get_logger

logger = get_logger(__name__)

# Maximum number of dept models to keep in memory simultaneously.
MAX_CACHED_MODELS = int(os.getenv("DEPT_LORA_CACHE_SIZE", "3"))
LOAD_TIMEOUT_SECONDS = int(os.getenv("DEPT_LORA_LOAD_TIMEOUT", "120"))
INFERENCE_TIMEOUT_SECONDS = int(os.getenv("DEPT_LORA_INFERENCE_TIMEOUT", "60"))

# ---------------------------------------------------------------------------
# Optional dependency check
# ---------------------------------------------------------------------------
try:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False
    logger.warning(
        "DeptLoRAService: peft / transformers not installed — "
        "LoRA inference will be unavailable. Install with: pip install peft transformers"
    )

try:
    from sft_experiment_manager import get_best_model_path_for_dept
    _SFT_MANAGER_AVAILABLE = True
except Exception as _exc:
    _SFT_MANAGER_AVAILABLE = False
    logger.warning("DeptLoRAService: sft_experiment_manager not available: %s", _exc)

    def get_best_model_path_for_dept(_dept: str):  # type: ignore[misc]
        return None


# ---------------------------------------------------------------------------
# DeptLoRAService
# ---------------------------------------------------------------------------

class DeptLoRAService:
    """Load, cache, and run inference with department-specific LoRA adapters."""

    def __init__(self) -> None:
        # OrderedDict used as an LRU cache: key=dept_name, value={"model", "tokenizer", "base_model_name"}
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()
        # Per-dept loading events prevent duplicate load threads
        self._loading: dict[str, threading.Event] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self, dept_name: str) -> bool:
        """Return True if a trained LoRA model exists on disk for *dept_name*."""
        if not _PEFT_AVAILABLE or not dept_name:
            logger.info("[PHASE2A] is_available(%r) → False (peft_available=%s)", dept_name, _PEFT_AVAILABLE)
            return False
        path = get_best_model_path_for_dept(dept_name)
        result = path is not None
        logger.info("[PHASE2A] is_available(%r) → %s  path=%r", dept_name, result, path)
        return result

    def generate(
        self,
        dept_name: str,
        query: str,
        max_new_tokens: int = 150,
    ) -> dict:
        """Generate a response using the dept-specific LoRA model.

        Returns a dict with keys:
        - ``success`` (bool)
        - ``response`` (str) — generated text
        - ``dept`` (str) — normalised dept name
        - ``source`` (str) — ``"lora"`` or ``"fallback"``
        - ``model_path`` (str) — path used
        - ``elapsed_seconds`` (float)
        - ``error`` (str, only on failure)
        """
        if not _PEFT_AVAILABLE:
            logger.warning("[PHASE2A] DeptLoRAService.generate: peft/transformers not installed")
            return {"success": False, "error": "peft/transformers not installed", "source": "fallback"}

        model_path = get_best_model_path_for_dept(dept_name)
        logger.info("[PHASE2A] DeptLoRAService.generate  dept=%r  model_path=%r", dept_name, model_path)
        if not model_path:
            logger.warning("[PHASE2A] DeptLoRAService.generate  NO MODEL FOUND for dept=%r", dept_name)
            return {
                "success": False,
                "error": f"No trained model found for department: {dept_name}",
                "source": "fallback",
            }

        loaded = self._get_or_load(dept_name, model_path)
        if not loaded:
            logger.error("[PHASE2A] DeptLoRAService.generate  LOAD FAILED for dept=%r", dept_name)
            return {
                "success": False,
                "error": f"Failed to load LoRA model for {dept_name}",
                "source": "fallback",
            }

        logger.info("[PHASE2A] DeptLoRAService.generate  model READY for dept=%r, running inference", dept_name)
        return self._run_inference(loaded, dept_name, model_path, query, max_new_tokens)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_load(self, dept_name: str, model_path: str) -> dict | None:
        """Return cached model dict, loading it if necessary (thread-safe)."""
        with self._lock:
            if dept_name in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(dept_name)
                logger.info("[PHASE2A] DeptLoRAService._get_or_load  CACHE HIT for dept=%r", dept_name)
                return self._cache[dept_name]

            # If another thread is already loading this dept, wait for it
            if dept_name in self._loading:
                logger.info("[PHASE2A] DeptLoRAService._get_or_load  WAITING — another thread is loading dept=%r", dept_name)
                event = self._loading[dept_name]
            else:
                logger.info("[PHASE2A] DeptLoRAService._get_or_load  CACHE MISS — starting load for dept=%r  path=%r", dept_name, model_path)
                event = threading.Event()
                self._loading[dept_name] = event
                # Kick off load in this thread (caller's thread; kept simple intentionally)
                loaded = self._load_model(dept_name, model_path)
                with self._lock:
                    if loaded:
                        self._evict_if_needed()
                        self._cache[dept_name] = loaded
                        logger.info("[PHASE2A] DeptLoRAService._get_or_load  CACHED dept=%r  cache_size=%d", dept_name, len(self._cache))
                    else:
                        logger.error("[PHASE2A] DeptLoRAService._get_or_load  LOAD RETURNED NONE for dept=%r", dept_name)
                    self._loading.pop(dept_name, None)
                    event.set()
                return loaded

        # Wait for the other thread to finish loading
        event.wait(timeout=LOAD_TIMEOUT_SECONDS)
        with self._lock:
            return self._cache.get(dept_name)

    def _load_model(self, dept_name: str, model_path: str) -> dict | None:
        """Load base model + LoRA adapter into memory.  Returns dict or None on failure."""
        logger.info("[PHASE2A] DeptLoRAService._load_model  START  dept=%r  path=%r  timeout=%ds",
                    dept_name, model_path, LOAD_TIMEOUT_SECONDS)

        # Read base model name from adapter_config.json
        base_model_name = self._read_base_model_name(model_path)
        logger.info("[PHASE2A] DeptLoRAService._load_model  base_model=%r", base_model_name)

        result_holder: list[dict | None] = [None]
        exc_holder: list[Exception | None] = [None]

        def _do_load():
            try:
                import torch as _torch
                from peft import PeftModel as _PeftModel
                from transformers import AutoModelForCausalLM as _AMCL, AutoTokenizer as _AT

                if _torch.cuda.is_available():
                    device = "cuda"
                    dtype = _torch.float16
                elif _torch.backends.mps.is_available():
                    device = "mps"
                    dtype = _torch.bfloat16
                else:
                    device = "cpu"
                    dtype = _torch.float32

                tokenizer = _AT.from_pretrained(model_path, trust_remote_code=True)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                base_model = _AMCL.from_pretrained(
                    base_model_name,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                    device_map="auto" if device == "cuda" else None,
                )
                model = _PeftModel.from_pretrained(base_model, model_path)
                model = model.merge_and_unload()
                if device != "cuda":
                    model = model.to(device)
                model.eval()
                model.config.use_cache = True

                result_holder[0] = {
                    "model": model,
                    "tokenizer": tokenizer,
                    "base_model_name": base_model_name,
                    "device": device,
                    "model_path": model_path,
                }
                logger.info("[PHASE2A] DeptLoRAService._load_model  SUCCESS  dept=%r  device=%s", dept_name, device)
            except Exception as exc:
                exc_holder[0] = exc
                logger.error("[PHASE2A] DeptLoRAService._load_model  FAILED  dept=%r  error=%s", dept_name, exc)

        t = threading.Thread(target=_do_load, daemon=True)
        t.start()
        logger.info("[PHASE2A] DeptLoRAService._load_model  thread started, waiting up to %ds …", LOAD_TIMEOUT_SECONDS)
        t.join(timeout=LOAD_TIMEOUT_SECONDS)

        if t.is_alive():
            logger.error(
                "[PHASE2A] DeptLoRAService._load_model  TIMEOUT after %ds  dept=%r",
                LOAD_TIMEOUT_SECONDS, dept_name,
            )
            return None

        if exc_holder[0]:
            return None

        logger.info("[PHASE2A] DeptLoRAService._load_model  DONE  dept=%r", dept_name)
        return result_holder[0]

    def _evict_if_needed(self) -> None:
        """Evict the LRU model if cache is at capacity.  Must be called with self._lock held."""
        while len(self._cache) >= MAX_CACHED_MODELS:
            evicted_dept, evicted = self._cache.popitem(last=False)
            # Best-effort cleanup to release GPU/CPU memory
            try:
                del evicted["model"]
                del evicted["tokenizer"]
                import gc
                gc.collect()
                try:
                    import torch as _torch
                    if _torch.cuda.is_available():
                        _torch.cuda.empty_cache()
                except Exception:
                    pass
            except Exception:
                pass
            logger.info("DeptLoRAService: evicted cached model for %s", evicted_dept)

    def _run_inference(
        self,
        loaded: dict,
        dept_name: str,
        model_path: str,
        query: str,
        max_new_tokens: int,
    ) -> dict:
        """Run generation with the already-loaded model dict, guarded by INFERENCE_TIMEOUT_SECONDS."""
        import re as _re
        import torch as _torch

        start = time.time()

        result_holder: list[dict | None] = [None]
        exc_holder: list[Exception | None] = [None]

        def _do_generate():
            try:
                model = loaded["model"]
                tokenizer = loaded["tokenizer"]
                device = loaded["device"]

                logger.info(
                    "[PHASE2A] DeptLoRAService._run_inference  START  dept=%r  device=%s  max_new_tokens=%d",
                    dept_name, device, max_new_tokens,
                )

                system_prompt = (
                    "You are a medical specialist in "
                    f"{dept_name}. "
                    "Provide clear, evidence-based clinical guidance appropriate for a healthcare professional. "
                    "Answer concisely in plain prose. "
                    "Do NOT include code, programming examples, markdown headers, or tutorial content."
                )
                prompt = (
                    f"### System:\n{system_prompt}\n\n"
                    f"### User:\n{query}\n\n"
                    f"### Assistant:\n"
                )

                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

                with _torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                response = tokenizer.decode(
                    output_ids[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                ).strip()

                # Strip code artifacts
                response = _re.sub(r'```[\s\S]*?```', '', response)
                response = _re.sub(r'`[^`]+`', '', response)
                response = _re.sub(
                    r'\n*(#+\s*)?(Exercise|Ideas?:|Solution:|Example[:\s]|import |def |print\()[^\n]*(\n[^\n]+)*',
                    '', response, flags=_re.IGNORECASE,
                )
                response = _re.sub(r'\n{3,}', '\n\n', response).strip()

                elapsed = round(time.time() - start, 2)
                tokens_generated = len(output_ids[0]) - inputs["input_ids"].shape[1]

                logger.info(
                    "[PHASE2A] DeptLoRAService._run_inference  DONE  dept=%r  tokens=%d  elapsed=%ss  response_len=%d",
                    dept_name, tokens_generated, elapsed, len(response),
                )

                result_holder[0] = {
                    "success": True,
                    "response": response,
                    "dept": dept_name,
                    "source": "lora",
                    "model_path": model_path,
                    "device": device,
                    "tokens_generated": int(tokens_generated),
                    "elapsed_seconds": elapsed,
                }
            except Exception as exc:
                exc_holder[0] = exc
                logger.error(
                    "[PHASE2A] DeptLoRAService._run_inference  EXCEPTION  dept=%r  error=%s",
                    dept_name, exc,
                )

        t = threading.Thread(target=_do_generate, daemon=True)
        t.start()
        t.join(timeout=INFERENCE_TIMEOUT_SECONDS)

        if t.is_alive():
            elapsed = round(time.time() - start, 2)
            logger.error(
                "[PHASE2A] DeptLoRAService._run_inference  TIMEOUT after %ds  dept=%r",
                INFERENCE_TIMEOUT_SECONDS, dept_name,
            )
            return {
                "success": False,
                "error": f"LoRA inference timed out after {INFERENCE_TIMEOUT_SECONDS}s",
                "dept": dept_name,
                "source": "fallback",
                "elapsed_seconds": elapsed,
            }

        if exc_holder[0]:
            return {
                "success": False,
                "error": str(exc_holder[0]),
                "dept": dept_name,
                "source": "fallback",
                "elapsed_seconds": round(time.time() - start, 2),
            }

        return result_holder[0] or {
            "success": False,
            "error": "inference thread returned no result",
            "dept": dept_name,
            "source": "fallback",
            "elapsed_seconds": round(time.time() - start, 2),
        }

    @staticmethod
    def _read_base_model_name(model_path: str) -> str:
        """Read base_model_name_or_path from adapter_config.json, or fall back to phi-2."""
        try:
            import json as _json
            cfg_path = os.path.join(model_path, "adapter_config.json")
            with open(cfg_path) as fh:
                cfg = _json.load(fh)
            return cfg.get("base_model_name_or_path", "microsoft/phi-2")
        except Exception:
            return "microsoft/phi-2"


# ---------------------------------------------------------------------------
# Module-level singleton (optional — main.py creates its own instance)
# ---------------------------------------------------------------------------
dept_lora_service = DeptLoRAService()

__all__ = ["DeptLoRAService", "dept_lora_service"]
