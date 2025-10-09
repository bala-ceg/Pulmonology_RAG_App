"""
RAG Architecture with Two-Store System and Lexical Gate
======================================================

This module implements a two-store RAG architecture with:
1. kb_local: internal PDFs + URLs 
2. kb_external: Wikipedia + arXiv
3. TF-IDF lexical gate for intelligent query routing
"""

import os
import json
import pickle
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WikipediaLoader, ArxivLoader


class TFIDFLexicalGate:
    """
    TF-IDF based lexical gate to decide query routing between local and external knowledge bases.
    """
    
    def __init__(self, threshold: float = 0.3):
        """
        Initialize the lexical gate.
        
        Args:
            threshold: TF-IDF similarity threshold for routing decisions
        """
        self.threshold = threshold
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2)
        )
        self.local_tfidf_matrix = None
        self.local_documents = []
        self.is_fitted = False
        
    def build_automation_summary(self, local_chunks: List[str]) -> None:
        """
        Build TF-IDF automation summary from local knowledge base chunks.
        
        Args:
            local_chunks: List of text chunks from local knowledge base
        """
        if not local_chunks:
            print("Warning: No local chunks provided for automation summary")
            return
            
        self.local_documents = local_chunks
        try:
            self.local_tfidf_matrix = self.vectorizer.fit_transform(local_chunks)
            self.is_fitted = True
            print(f"Built automation summary from {len(local_chunks)} local chunks")
        except Exception as e:
            print(f"Error building automation summary: {e}")
            self.is_fitted = False
    
    def should_query_local_first(self, query: str) -> Tuple[bool, float]:
        """
        Determine if query should hit local KB first based on TF-IDF similarity.
        
        Args:
            query: User query string
            
        Returns:
            Tuple of (should_query_local_first, max_similarity_score)
        """
        if not self.is_fitted:
            # If no local knowledge, always go external first
            return False, 0.0
            
        try:
            # Transform query using the fitted vectorizer
            query_tfidf = self.vectorizer.transform([query])
            
            # Calculate similarity with local documents
            similarities = cosine_similarity(query_tfidf, self.local_tfidf_matrix).flatten()
            max_similarity = float(np.max(similarities)) if len(similarities) > 0 else 0.0
            
            # Decide routing based on threshold
            query_local_first = bool(max_similarity >= self.threshold)
            
            print(f"TF-IDF Gate: max_similarity={max_similarity:.3f}, threshold={self.threshold}, local_first={query_local_first}")
            
            return query_local_first, max_similarity
            
        except Exception as e:
            print(f"Error in lexical gate routing: {e}")
            return False, 0.0
    
    def save_to_disk(self, filepath: str) -> None:
        """Save the fitted vectorizer and TF-IDF matrix to disk."""
        if not self.is_fitted:
            print("Warning: Cannot save unfitted lexical gate")
            return
            
        try:
            gate_data = {
                'vectorizer': self.vectorizer,
                'local_tfidf_matrix': self.local_tfidf_matrix,
                'local_documents': self.local_documents,
                'threshold': self.threshold,
                'is_fitted': self.is_fitted
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(gate_data, f)
                
            print(f"Lexical gate saved to {filepath}")
            
        except Exception as e:
            print(f"Error saving lexical gate: {e}")
    
    def load_from_disk(self, filepath: str) -> bool:
        """Load the fitted vectorizer and TF-IDF matrix from disk."""
        try:
            if not os.path.exists(filepath):
                print(f"Lexical gate file not found: {filepath}")
                return False
                
            with open(filepath, 'rb') as f:
                gate_data = pickle.load(f)
                
            self.vectorizer = gate_data['vectorizer']
            self.local_tfidf_matrix = gate_data['local_tfidf_matrix']
            self.local_documents = gate_data['local_documents']
            self.threshold = gate_data['threshold']
            self.is_fitted = gate_data['is_fitted']
            
            print(f"Lexical gate loaded from {filepath}")
            return True
            
        except Exception as e:
            print(f"Error loading lexical gate: {e}")
            return False


class TwoStoreRAGManager:
    """
    Manager for the two-store RAG architecture with intelligent routing.
    """
    
    def __init__(self, embeddings, llm, base_vector_path: str = "./vector_dbs"):
        """
        Initialize the RAG manager.
        
        Args:
            embeddings: OpenAI embeddings instance
            llm: Language model instance
            base_vector_path: Base path for vector databases
        """
        self.embeddings = embeddings
        self.llm = llm
        self.base_vector_path = base_vector_path
        
        # Vector store paths
        self.kb_local_path = os.path.join(base_vector_path, "kb_local")
        self.kb_external_path = os.path.join(base_vector_path, "kb_external")
        self.lexical_gate_path = os.path.join(base_vector_path, "lexical_gate.pkl")
        
        # Create directories
        os.makedirs(self.kb_local_path, exist_ok=True)
        os.makedirs(self.kb_external_path, exist_ok=True)
        
        # Initialize components
        self.kb_local = None
        self.kb_external = None
        self.lexical_gate = TFIDFLexicalGate()
        
        # Session-specific vector database cache
        self.session_cache = {}
        self.current_session_id = None
        
        # Load existing components (only external KB and lexical gate)
        self._initialize_external_vector_store()
        self.lexical_gate.load_from_disk(self.lexical_gate_path)
    
    def _initialize_external_vector_store(self):
        """Initialize or load existing external vector store only."""
        try:
            # Initialize kb_external  
            if os.path.exists(self.kb_external_path) and os.listdir(self.kb_external_path):
                self.kb_external = Chroma(persist_directory=self.kb_external_path, embedding_function=self.embeddings)
                print("Loaded existing kb_external")
            else:
                print("kb_external not found - will be created when documents are added")
                
        except Exception as e:
            print(f"Error initializing external vector store: {e}")
    
    def load_session_vector_db(self, session_id: str) -> bool:
        """Load session-specific vector database dynamically.
        
        Args:
            session_id: Session identifier for the vector database
            
        Returns:
            True if session vector DB was loaded successfully, False otherwise
        """
        try:
            # If already loaded for this session, return True
            if self.current_session_id == session_id and self.kb_local is not None:
                return True
            
            # Check if session vector DB exists
            session_vector_path = os.path.join(self.base_vector_path, session_id)
            
            if not os.path.exists(session_vector_path) or not os.listdir(session_vector_path):
                print(f"No vector DB found for session {session_id}")
                self.kb_local = None
                self.current_session_id = None
                return False
            
            # Check cache first
            if session_id in self.session_cache:
                self.kb_local = self.session_cache[session_id]
                self.current_session_id = session_id
                print(f"Loaded session vector DB from cache: {session_id}")
                return True
            
            # Load session-specific vector database
            self.kb_local = Chroma(persist_directory=session_vector_path, embedding_function=self.embeddings)
            
            # Cache the loaded vector DB (limit cache size to prevent memory issues)
            if len(self.session_cache) >= 5:  # Limit cache to 5 sessions
                # Remove oldest entry
                oldest_session = next(iter(self.session_cache))
                del self.session_cache[oldest_session]
            
            self.session_cache[session_id] = self.kb_local
            self.current_session_id = session_id
            
            print(f"Loaded session vector DB: {session_id}")
            return True
            
        except Exception as e:
            print(f"Error loading session vector DB for {session_id}: {e}")
            self.kb_local = None
            self.current_session_id = None
            return False
    
    def add_documents_to_local(self, documents: List[Document]) -> None:
        """
        Add documents to the local knowledge base and update the lexical gate.
        
        Args:
            documents: List of LangChain Document objects
        """
        try:
            if not documents:
                print("No documents to add to kb_local")
                return
                
            # Create or update kb_local
            if self.kb_local is None:
                self.kb_local = Chroma.from_documents(
                    documents, 
                    embedding=self.embeddings, 
                    persist_directory=self.kb_local_path
                )
                print(f"Created kb_local with {len(documents)} documents")
            else:
                self.kb_local.add_documents(documents)
                print(f"Added {len(documents)} documents to kb_local")
            
            # Update lexical gate with new content
            local_chunks = [doc.page_content for doc in documents]
            
            # If gate exists, get existing chunks and combine
            if self.lexical_gate.is_fitted:
                existing_chunks = self.lexical_gate.local_documents
                all_chunks = existing_chunks + local_chunks
            else:
                all_chunks = local_chunks
            
            self.lexical_gate.build_automation_summary(all_chunks)
            self.lexical_gate.save_to_disk(self.lexical_gate_path)
            
        except Exception as e:
            print(f"Error adding documents to kb_local: {e}")
    
    def add_documents_to_external(self, documents: List[Document]) -> None:
        """
        Add documents to the external knowledge base.
        
        Args:
            documents: List of LangChain Document objects
        """
        try:
            if not documents:
                print("No documents to add to kb_external")
                return
                
            # Create or update kb_external
            if self.kb_external is None:
                self.kb_external = Chroma.from_documents(
                    documents, 
                    embedding=self.embeddings, 
                    persist_directory=self.kb_external_path
                )
                print(f"Created kb_external with {len(documents)} documents")
            else:
                self.kb_external.add_documents(documents)
                print(f"Added {len(documents)} documents to kb_external")
                
        except Exception as e:
            print(f"Error adding documents to kb_external: {e}")
    
    def has_external_content(self) -> bool:
        """Check if external KB has any content."""
        if not self.kb_external:
            return False
        try:
            count = self.kb_external._collection.count()
            return count > 0
        except:
            return False
    
    def get_local_content_count(self) -> int:
        """Get the number of documents in local KB."""
        if not self.kb_local:
            return 0
        try:
            return self.kb_local._collection.count()
        except:
            return 0

    def get_external_content_count(self) -> int:
        """Get the number of documents in external KB."""
        if not self.kb_external:
            return 0
        try:
            return self.kb_external._collection.count()
        except:
            return 0

    def has_session_content(self, session_id: str) -> bool:
        """Check if a session has vector database content without loading it."""
        if not session_id:
            return False
            
        session_path = os.path.join(self.base_vector_path, session_id)
        if not os.path.exists(session_path):
            return False
            
        # Check if there's a chroma.sqlite3 file (indicates ChromaDB data)
        chroma_file = os.path.join(session_path, "chroma.sqlite3")
        if os.path.exists(chroma_file):
            # Quick check: file size > minimal threshold indicates content
            try:
                size = os.path.getsize(chroma_file)
                return size > 8192  # 8KB threshold for meaningful content
            except:
                return False
                
        return False

    def load_wikipedia_content(self, topics: List[str], max_docs_per_topic: int = 3, force_reload: bool = False) -> None:
        """
        Load Wikipedia content into kb_external.
        
        Args:
            topics: List of Wikipedia topics to load
            max_docs_per_topic: Maximum documents to load per topic
            force_reload: If True, load content even if external KB already has content
        """
        # Skip loading if external KB already has content and not forcing reload
        if not force_reload and self.has_external_content():
            count = self.get_external_content_count()
            print(f"📊 External KB already has {count} documents, skipping Wikipedia loading")
            print("   Use force_reload=True to add more content")
            return
            
        try:
            all_docs = []
            
            for topic in topics:
                print(f"Loading Wikipedia content for: {topic}")
                
                try:
                    loader = WikipediaLoader(query=topic, load_max_docs=max_docs_per_topic)
                    docs = loader.load()
                    
                    # Add source metadata
                    for doc in docs:
                        doc.metadata['source_type'] = 'wikipedia'
                        doc.metadata['topic'] = topic
                    
                    all_docs.extend(docs)
                    print(f"Loaded {len(docs)} Wikipedia documents for {topic}")
                    
                except Exception as e:
                    print(f"Error loading Wikipedia content for {topic}: {e}")
            
            if all_docs:
                # Split documents if they're too large
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                split_docs = text_splitter.split_documents(all_docs)
                
                self.add_documents_to_external(split_docs)
                print(f"Added {len(split_docs)} Wikipedia chunks to kb_external")
                
        except Exception as e:
            print(f"Error loading Wikipedia content: {e}")
    
    def load_arxiv_content(self, queries: List[str], max_docs_per_query: int = 2, force_reload: bool = False) -> None:
        """
        Load arXiv content into kb_external.
        
        Args:
            queries: List of arXiv search queries
            max_docs_per_query: Maximum documents to load per query
            force_reload: If True, load content even if external KB already has content
        """
        # Skip loading if external KB already has content and not forcing reload
        if not force_reload and self.has_external_content():
            count = self.get_external_content_count()
            print(f"📊 External KB already has {count} documents, skipping arXiv loading")
            print("   Use force_reload=True to add more content")
            return
            
        try:
            all_docs = []
            
            for query in queries:
                print(f"Loading arXiv content for: {query}")
                
                try:
                    loader = ArxivLoader(query=query, load_max_docs=max_docs_per_query)
                    docs = loader.load()
                    
                    # Add source metadata
                    for doc in docs:
                        doc.metadata['source_type'] = 'arxiv'
                        doc.metadata['query'] = query
                    
                    all_docs.extend(docs)
                    print(f"Loaded {len(docs)} arXiv documents for {query}")
                    
                except Exception as e:
                    print(f"Error loading arXiv content for {query}: {e}")
            
            if all_docs:
                # Split documents if they're too large
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                split_docs = text_splitter.split_documents(all_docs)
                
                self.add_documents_to_external(split_docs)
                print(f"Added {len(split_docs)} arXiv chunks to kb_external")
                
        except Exception as e:
            print(f"Error loading arXiv content: {e}")
    
    def query_with_routing(self, query: str, session_id: str = None) -> Dict:
        """
        Query the RAG system with intelligent routing based on the lexical gate.
        
        Args:
            query: User query string
            session_id: Session identifier for loading appropriate vector database
            
        Returns:
            Dictionary with response, citations, and routing information
        """
        try:
            # Load session-specific vector database if session_id provided
            session_kb_loaded = False
            if session_id:
                session_kb_loaded = self.load_session_vector_db(session_id)
            
            # Use lexical gate to determine routing
            query_local_first, similarity_score = self.lexical_gate.should_query_local_first(query)
            
            responses = []
            citations = []
            routing_info = {
                'similarity_score': similarity_score,
                'query_local_first': query_local_first,
                'sources_queried': [],
                'session_id': session_id,
                'session_kb_loaded': session_kb_loaded
            }
            
            if query_local_first:
                # Query kb_local first, fallback to kb_external if weak
                local_response = self._query_kb_local(query)
                if local_response and self._is_strong_response(local_response['result']):
                    responses.append({
                        'source': 'Adhoc Documents',
                        'content': local_response['result'],
                        'confidence': 90
                    })
                    citations.extend(local_response.get('citations', []))
                    routing_info['sources_queried'].append('kb_local')
                else:
                    # Weak local response, try external
                    external_response = self._query_kb_external(query)
                    if external_response:
                        responses.append({
                            'source': 'Third Party Research',
                            'content': external_response['result'],
                            'confidence': 70
                        })
                        citations.extend(external_response.get('citations', []))
                        routing_info['sources_queried'].extend(['kb_local_weak', 'kb_external'])
                    else:
                        # Include weak local response if external also fails
                        if local_response:
                            responses.append({
                                'source': 'Adhoc Documents',
                                'content': local_response['result'],
                                'confidence': 60
                            })
                            citations.extend(local_response.get('citations', []))
                            routing_info['sources_queried'].append('kb_local_weak')
            else:
                # Query kb_external first, fallback to kb_local if weak
                external_response = self._query_kb_external(query)
                if external_response and self._is_strong_response(external_response['result']):
                    responses.append({
                        'source': 'Third Party Research',
                        'content': external_response['result'],
                        'confidence': 85
                    })
                    citations.extend(external_response.get('citations', []))
                    routing_info['sources_queried'].append('kb_external')
                else:
                    # Weak external response, try local
                    local_response = self._query_kb_local(query)
                    if local_response:
                        responses.append({
                            'source': 'Adhoc Documents',
                            'content': local_response['result'],
                            'confidence': 75
                        })
                        citations.extend(local_response.get('citations', []))
                        routing_info['sources_queried'].extend(['kb_external_weak', 'kb_local'])
                    else:
                        # Include weak external response if local also fails
                        if external_response:
                            responses.append({
                                'source': 'Third Party Research',
                                'content': external_response['result'],
                                'confidence': 50
                            })
                            citations.extend(external_response.get('citations', []))
                            routing_info['sources_queried'].append('kb_external_weak')
            
            return {
                'responses': responses,
                'citations': citations,
                'routing_info': routing_info
            }
            
        except Exception as e:
            print(f"Error in query routing: {e}")
            return {
                'responses': [],
                'citations': [],
                'routing_info': {'error': str(e)}
            }
    
    def guarded_retrieve(self, query: str, retriever, similarity_threshold: float = 0.35) -> Optional[List]:
        """
        Post-retrieval guard that checks similarity and content relevance.
        
        Args:
            query: User query string
            retriever: LangChain retriever object
            similarity_threshold: Minimum average similarity score
            
        Returns:
            Documents if they pass the guard, None if they should trigger fallback
        """
        try:
            # Retrieve documents
            docs = retriever.invoke(query)
            
            if not docs:
                return None
            
            # Extract main query terms for relevance checking
            import re
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'what', 'how', 'why', 'when', 'where'}
            cleaned_query = re.sub(r'[^\w\s]', '', query.lower())
            main_terms = [word for word in cleaned_query.split() if word not in stop_words and len(word) > 2]
            
            if not main_terms:
                # If we can't extract terms, use simple length check
                if all(len(doc.page_content.strip()) < 50 for doc in docs):
                    return None
                return docs
            
            # Check if any of the top chunks contain main nouns
            relevant_docs = []
            for doc in docs:
                content_lower = doc.page_content.lower()
                if any(term in content_lower for term in main_terms):
                    relevant_docs.append(doc)
            
            # If no documents contain main terms, trigger fallback
            if not relevant_docs:
                print(f"Guard: No documents contain main terms {main_terms} - triggering fallback")
                return None
            
            # Check document quality - avoid very short or generic responses
            quality_docs = []
            for doc in relevant_docs:
                content = doc.page_content.strip()
                if len(content) > 50 and not self._is_generic_content(content):
                    quality_docs.append(doc)
            
            if not quality_docs:
                print("Guard: No quality documents found - triggering fallback")
                return None
            
            return quality_docs
            
        except Exception as e:
            print(f"Error in guarded_retrieve: {e}")
            return None
    
    def _is_generic_content(self, content: str) -> bool:
        """Check if content is too generic or unhelpful."""
        generic_indicators = [
            "no information available",
            "not found in the context",
            "insufficient data",
            "cannot determine",
            "more information needed"
        ]
        
        content_lower = content.lower()
        return any(indicator in content_lower for indicator in generic_indicators)

    def _query_kb_local(self, query: str) -> Optional[Dict]:
        """Query the local knowledge base with guarded retrieval."""
        try:
            if self.kb_local is None:
                return None
                
            retriever = self.kb_local.as_retriever(search_type="similarity", search_kwargs={"k": 5})
            
            # Apply guarded retrieval
            search_results = self.guarded_retrieve(query, retriever)
            
            if search_results is None:
                print("Local KB query failed guard check - low relevance")
                return None
            
            # Use the filtered results for QA chain
            qa_chain = RetrievalQA.from_chain_type(llm=self.llm, retriever=retriever)
            response = qa_chain.invoke(query)
            
            return {
                'result': response['result'],
                'citations': self._format_citations(search_results, 'Local KB'),
                'guard_passed': True
            }
            
        except Exception as e:
            print(f"Error querying kb_local: {e}")
            return None
    
    def _query_kb_external(self, query: str) -> Optional[Dict]:
        """Query the external knowledge base with guarded retrieval."""
        try:
            if self.kb_external is None:
                return None
                
            retriever = self.kb_external.as_retriever(search_type="similarity", search_kwargs={"k": 5})
            
            # Apply guarded retrieval
            search_results = self.guarded_retrieve(query, retriever)
            
            if search_results is None:
                print("External KB query failed guard check - low relevance")
                return None
            
            # Use the filtered results for QA chain
            qa_chain = RetrievalQA.from_chain_type(llm=self.llm, retriever=retriever)
            response = qa_chain.invoke(query)
            
            return {
                'result': response['result'],
                'citations': self._format_citations(search_results, 'External KB'),
                'guard_passed': True
            }
            
        except Exception as e:
            print(f"Error querying kb_external: {e}")
            return None
    
    def _is_strong_response(self, response: str) -> bool:
        """
        Determine if a response is strong enough to avoid fallback.
        
        Args:
            response: LLM response string
            
        Returns:
            True if response is considered strong
        """
        if not response or len(response.strip()) < 50:
            return False
            
        # Check for common weak response indicators
        weak_indicators = [
            "I don't know",
            "I'm not sure",
            "I don't have information",
            "I cannot find",
            "insufficient information",
            "not available in the provided",
            "based on the provided context, I cannot"
        ]
        
        response_lower = response.lower()
        for indicator in weak_indicators:
            if indicator in response_lower:
                return False
                
        return True
    
    def _format_citations(self, search_results: List, source_prefix: str) -> List[str]:
        """Format search results into citation strings with user-friendly display names."""
        citations = []
        
        # Map technical source prefixes to user-friendly display names
        display_names = {
            'Local KB': 'Adhoc Documents',
            'External KB': 'Third Party Research'
        }
        
        # Use display name if available, otherwise use original prefix
        display_prefix = display_names.get(source_prefix, source_prefix)
        
        try:
            for i, doc in enumerate(search_results, 1):
                metadata = doc.metadata
                
                # Format citation based on source type
                if metadata.get('source_type') == 'wikipedia':
                    citation = f"**{display_prefix} - Wikipedia**: {metadata.get('title', 'Unknown')} - {metadata.get('source', 'N/A')}"
                elif metadata.get('source_type') == 'arxiv':
                    citation = f"**{display_prefix} - arXiv**: {metadata.get('Title', 'Unknown')} by {metadata.get('Authors', 'Unknown')} - {metadata.get('source', 'N/A')}"
                else:
                    # Local files
                    source = metadata.get('source', 'Unknown')
                    citation = f"**{display_prefix}**: {os.path.basename(source) if source != 'Unknown' else 'Unknown Source'}"
                
                citations.append(citation)
                
        except Exception as e:
            print(f"Error formatting citations: {e}")
            
        return citations


class MedicalQueryRouter:
    """
    Intelligent query router that selects appropriate tools based on query analysis.
    """
    
    def __init__(self, rag_manager=None):
        """
        Initialize the medical query router.
        
        Args:
            rag_manager: TwoStoreRAGManager instance for checking content availability
        """
        self.rag_manager = rag_manager
        
        # Keywords for tool selection heuristics
        self.arxiv_keywords = [
            'latest', 'recent', 'new', 'research', 'study', 'paper', 'findings',
            'experiment', 'trial', 'investigation', 'preprint', 'published',
            'scientific', 'evidence', 'current research', 'recent developments',
            'newest', 'breakthrough', 'cutting-edge', 'novel', 'innovative'
        ]
        
        self.internal_keywords = [
            'uploaded', 'my file', 'my document', 'my pdf', 'uploaded document',
            'our protocol', 'our guideline', 'organization', 'my organization',
            'internal', 'company', 'our study', 'my study', 'this document',
            'uploaded content', 'my data', 'our data', 'session'
        ]
        
        self.wikipedia_keywords = [
            'what is', 'define', 'definition', 'explain', 'tell me about',
            'overview', 'background', 'basic', 'general', 'introduction',
            'simple explanation', 'layman', 'understand', 'meaning'
        ]
    
    def route_tools(self, query: str, session_id: str = None) -> Dict:
        """
        Route query to appropriate tools with ranking and confidence scoring.
        
        Args:
            query: User query string
            session_id: Session identifier for checking user-specific content
            
        Returns:
            Dictionary with ranked tools, confidence, and reasoning
        """
        try:
            query_lower = query.lower()
            
            # Initialize scores for each tool
            tool_scores = {
                'Wikipedia_Search': 0,
                'ArXiv_Search': 0,
                'Internal_VectorDB': 0
            }
            
            # Keyword-based scoring
            for keyword in self.arxiv_keywords:
                if keyword in query_lower:
                    tool_scores['ArXiv_Search'] += 2
                    
            for keyword in self.internal_keywords:
                if keyword in query_lower:
                    tool_scores['Internal_VectorDB'] += 3
                    
            for keyword in self.wikipedia_keywords:
                if keyword in query_lower:
                    tool_scores['Wikipedia_Search'] += 2
            
            # Context-based adjustments - QUERY-CONTENT-AWARE ROUTING
            # Check if session has content (without loading it)
            has_session_content = (session_id and 
                                 self.rag_manager and 
                                 self.rag_manager.has_session_content(session_id))
            
            has_external_content = (self.rag_manager and 
                                  self.rag_manager.has_external_content())
            
            # Calculate content relevance scores instead of unconditional boosting
            pdf_relevance_score = self._calculate_pdf_relevance(query_lower)
            wiki_relevance_score = self._calculate_wiki_relevance(query_lower)
            arxiv_relevance_score = self._calculate_arxiv_relevance(query_lower)
            
            # Apply relevance-based scoring  
            tool_scores['Internal_VectorDB'] += pdf_relevance_score
            tool_scores['Wikipedia_Search'] += wiki_relevance_score  
            tool_scores['ArXiv_Search'] += arxiv_relevance_score
            
            # Debug logging for routing decision factors
            print(f"🔍 Routing Analysis for: '{query}'")
            print(f"📊 Relevance Scores - PDF: {pdf_relevance_score}, Wiki: {wiki_relevance_score}, ArXiv: {arxiv_relevance_score}")
            print(f"📁 Session Context - ID: {session_id}, Has Content: {has_session_content}")
            print(f"🎯 Current Tool Scores - {tool_scores}")
            
            # Only boost Internal_VectorDB if session has content AND query is relevant to PDF content
            if has_session_content and pdf_relevance_score > 0:
                tool_scores['Internal_VectorDB'] += 1  # Modest boost for relevant session content
                print(f"🎯 Boosting Internal_VectorDB: Session content available and query relevant (session: {session_id})")
            elif has_session_content and pdf_relevance_score == 0:
                # Session has content but query isn't PDF-relevant
                print(f"📄 Session {session_id} has content but query not PDF-relevant - allowing content-based routing")
            elif session_id:
                # Session provided but has no content
                print(f"⚠️  Session {session_id} has no content - routing to external sources")
                tool_scores['Internal_VectorDB'] = max(0, tool_scores['Internal_VectorDB'] - 2)
            else:
                # No session provided - prefer external sources for general queries
                print(f"ℹ️  No session provided - routing to external sources")
                tool_scores['Internal_VectorDB'] = max(0, tool_scores['Internal_VectorDB'] - 1)
            
            # Default scoring if no keywords match
            if max(tool_scores.values()) == 0:
                # Session-aware default hierarchy
                if has_session_content:
                    # Prioritize internal content when user has session-specific documents
                    tool_scores['Internal_VectorDB'] = 4
                    tool_scores['Wikipedia_Search'] = 3
                    tool_scores['ArXiv_Search'] = 2
                else:
                    # Standard hierarchy when no session content
                    tool_scores['Wikipedia_Search'] = 3
                    tool_scores['ArXiv_Search'] = 2
                    tool_scores['Internal_VectorDB'] = 0
            
            # Sort tools by score (descending)
            ranked_tools = sorted(tool_scores.items(), 
                                key=lambda x: x[1], 
                                reverse=True)
            
            # Calculate confidence based on score differences
            top_score = ranked_tools[0][1]
            second_score = ranked_tools[1][1] if len(ranked_tools) > 1 else 0
            
            if top_score <= 0:
                confidence = 'low'
                confidence_score = 0.3
            elif top_score - second_score >= 3:
                confidence = 'high'
                confidence_score = 0.9
            elif top_score - second_score >= 1:
                confidence = 'medium'
                confidence_score = 0.7
            else:
                confidence = 'low'
                confidence_score = 0.4
            
            # Generate reasoning
            primary_tool = ranked_tools[0][0]
            reasoning = self._generate_reasoning(query, primary_tool, tool_scores, 
                                               has_session_content, has_external_content)
            
            # Final routing decision logging
            print(f"🏆 Final Routing Decision: {primary_tool} (confidence: {confidence})")
            print(f"💭 Reasoning: {reasoning}")
            print(f"📋 Ranked Tools: {[tool for tool, score in ranked_tools if score > 0]}")
            
            # Return top 1-2 tools as requested
            selected_tools = [tool for tool, score in ranked_tools[:2] if score > 0]
            
            return {
                'ranked_tools': selected_tools,
                'primary_tool': primary_tool,
                'confidence': confidence,
                'confidence_score': confidence_score,
                'reasoning': reasoning,
                'tool_scores': tool_scores,
                'session_has_content': has_session_content
            }
            
        except Exception as e:
            print(f"Error in route_tools: {e}")
            return {
                'ranked_tools': ['Wikipedia_Search'],
                'primary_tool': 'Wikipedia_Search',
                'confidence': 'low',
                'confidence_score': 0.3,
                'reasoning': f"Error in routing, defaulting to Wikipedia: {str(e)}",
                'tool_scores': {'Wikipedia_Search': 1, 'ArXiv_Search': 0, 'Internal_VectorDB': 0},
                'session_has_content': False
            }
    
    def _generate_reasoning(self, query: str, primary_tool: str, scores: Dict, 
                          has_local: bool, has_external: bool) -> str:
        """Generate human-readable reasoning for tool selection."""
        
        query_lower = query.lower()
        reasons = []
        
        if primary_tool == 'ArXiv_Search':
            if any(kw in query_lower for kw in ['latest', 'recent', 'research', 'study']):
                reasons.append("Query contains research-oriented keywords")
            reasons.append("ArXiv selected for scientific papers and recent findings")
        
        elif primary_tool == 'Internal_VectorDB':
            if any(kw in query_lower for kw in ['uploaded', 'my', 'our']):
                reasons.append("Query references user-specific or uploaded content")
            if has_local:
                reasons.append("User has uploaded documents available")
            else:
                reasons.append("WARNING: Internal documents requested but none available")
        
        elif primary_tool == 'Wikipedia_Search':
            if any(kw in query_lower for kw in ['what is', 'define', 'explain']):
                reasons.append("Query seeks definitions or general explanations")
            reasons.append("Wikipedia selected for encyclopedic knowledge")
        
        # Add confidence reasoning
        score_diff = scores[primary_tool] - max([s for k, s in scores.items() if k != primary_tool])
        if score_diff >= 3:
            reasons.append("High confidence in tool selection based on keyword analysis")
        elif score_diff >= 1:
            reasons.append("Medium confidence - clear preference but alternatives possible")
        else:
            reasons.append("Low confidence - multiple tools could be relevant")
        
        return "; ".join(reasons) if reasons else "Default selection based on general query pattern"
    
    def _calculate_pdf_relevance(self, query_lower: str) -> int:
        """Calculate relevance score for PDF/Internal content based on query terms."""
        pdf_indicators = [
            'rdw', 'red blood cell distribution', 'mortality', 'copd', 'patients', 'study',
            'analysis', 'cohort', 'clinical', 'medical research', 'hospital', 'treatment',
            'diagnosis', 'outcome', 'statistical', 'regression', 'correlation', 'odds ratio',
            'confidence interval', 'p value', 'significant', 'investigation', 'findings',
            'results', 'conclusion', 'method', 'participant', 'baseline', 'characteristic'
        ]
        
        score = 0
        for indicator in pdf_indicators:
            if indicator in query_lower:
                score += 2
        
        # Additional scoring for medical/research patterns
        if any(pattern in query_lower for pattern in ['what was the relationship', 'increasing', 'levels']):
            score += 3
        if any(pattern in query_lower for pattern in ['between', 'and', 'association']):
            score += 2
            
        return min(score, 8)  # Cap at 8 points
    
    def _calculate_wiki_relevance(self, query_lower: str) -> int:
        """Calculate relevance score for Wikipedia content based on query terms."""
        wiki_indicators = [
            'what are', 'symptoms', 'covid19', 'coronavirus', 'definition', 'overview',
            'explain', 'tell me about', 'basic', 'general', 'introduction', 'meaning',
            'cause', 'prevention', 'treatment options', 'types of', 'common', 'typical'
        ]
        
        score = 0
        for indicator in wiki_indicators:
            if indicator in query_lower:
                score += 2
                
        # Boost for general knowledge queries  
        if query_lower.startswith(('what', 'how', 'why', 'when', 'where')):
            score += 1
            
        return min(score, 6)  # Cap at 6 points
    
    def _calculate_arxiv_relevance(self, query_lower: str) -> int:
        """Calculate relevance score for ArXiv content based on query terms."""
        arxiv_indicators = [
            'latest research', 'recent', 'new', 'research', 'study', 'paper', 'findings',  
            'experiment', 'trial', 'investigation', 'preprint', 'published', 'scientific',
            'evidence', 'current research', 'recent developments', 'newest', 'breakthrough',
            'cutting-edge', 'novel', 'innovative', 'pulmonology', 'get me the latest'
        ]
        
        score = 0 
        for indicator in arxiv_indicators:
            if indicator in query_lower:
                score += 2
                
        # Strong boost for explicit research requests
        if 'latest' in query_lower and 'research' in query_lower:
            score += 3
            
        return min(score, 8)  # Cap at 8 points