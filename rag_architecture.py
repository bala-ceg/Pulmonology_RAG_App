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
        
        # Load existing components
        self._initialize_vector_stores()
        self.lexical_gate.load_from_disk(self.lexical_gate_path)
    
    def _initialize_vector_stores(self):
        """Initialize or load existing vector stores."""
        try:
            # Initialize kb_local
            if os.path.exists(self.kb_local_path) and os.listdir(self.kb_local_path):
                self.kb_local = Chroma(persist_directory=self.kb_local_path, embedding_function=self.embeddings)
                print("Loaded existing kb_local")
            else:
                print("kb_local not found - will be created when documents are added")
            
            # Initialize kb_external  
            if os.path.exists(self.kb_external_path) and os.listdir(self.kb_external_path):
                self.kb_external = Chroma(persist_directory=self.kb_external_path, embedding_function=self.embeddings)
                print("Loaded existing kb_external")
            else:
                print("kb_external not found - will be created when documents are added")
                
        except Exception as e:
            print(f"Error initializing vector stores: {e}")
    
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
    
    def query_with_routing(self, query: str) -> Dict:
        """
        Query the RAG system with intelligent routing based on the lexical gate.
        
        Args:
            query: User query string
            
        Returns:
            Dictionary with response, citations, and routing information
        """
        try:
            # Use lexical gate to determine routing
            query_local_first, similarity_score = self.lexical_gate.should_query_local_first(query)
            
            responses = []
            citations = []
            routing_info = {
                'similarity_score': similarity_score,
                'query_local_first': query_local_first,
                'sources_queried': []
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
    
    def _query_kb_local(self, query: str) -> Optional[Dict]:
        """Query the local knowledge base."""
        try:
            if self.kb_local is None:
                return None
                
            retriever = self.kb_local.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            qa_chain = RetrievalQA.from_chain_type(llm=self.llm, retriever=retriever)
            
            response = qa_chain.invoke(query)
            search_results = retriever.invoke(query)
            
            return {
                'result': response['result'],
                'citations': self._format_citations(search_results, 'Local KB')
            }
            
        except Exception as e:
            print(f"Error querying kb_local: {e}")
            return None
    
    def _query_kb_external(self, query: str) -> Optional[Dict]:
        """Query the external knowledge base."""
        try:
            if self.kb_external is None:
                return None
                
            retriever = self.kb_external.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            qa_chain = RetrievalQA.from_chain_type(llm=self.llm, retriever=retriever)
            
            response = qa_chain.invoke(query)
            search_results = retriever.invoke(query)
            
            return {
                'result': response['result'],
                'citations': self._format_citations(search_results, 'External KB')
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