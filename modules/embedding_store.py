import os
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import json
import time
import traceback
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingStore:
    """
    Simple in-memory store for document embeddings.
    Provides similarity search functionality without requiring a database.
    """
    
    def __init__(self, config=None):
        """
        Initialize the EmbeddingStore
        
        Args:
            config: Configuration dictionary
        """
        print("[DEBUG] Initializing EmbeddingStore")
        self.config = config or {}
        self.documents = []
        self.embeddings = None
        
        # Create data directory if it doesn't exist
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load existing embeddings if available
        self._load_embeddings()
        
        print(f"[DEBUG] EmbeddingStore initialized with {len(self.documents)} documents")
    
    def _load_embeddings(self):
        """Load embeddings from disk if available"""
        embeddings_path = os.path.join(self.data_dir, "embeddings.npy")
        documents_path = os.path.join(self.data_dir, "documents.json")
        
        if os.path.exists(embeddings_path) and os.path.exists(documents_path):
            try:
                self.embeddings = np.load(embeddings_path)
                with open(documents_path, 'r') as f:
                    self.documents = json.load(f)
                print(f"[DEBUG] Loaded {len(self.documents)} documents and embeddings from disk")
            except Exception as e:
                print(f"[ERROR] Error loading embeddings: {e}")
                self.documents = []
                self.embeddings = None
    
    def _save_embeddings(self):
        """Save embeddings to disk"""
        if self.embeddings is not None and len(self.documents) > 0:
            try:
                embeddings_path = os.path.join(self.data_dir, "embeddings.npy")
                documents_path = os.path.join(self.data_dir, "documents.json")
                
                np.save(embeddings_path, self.embeddings)
                with open(documents_path, 'w') as f:
                    json.dump(self.documents, f)
                
                print(f"[DEBUG] Saved {len(self.documents)} documents and embeddings to disk")
            except Exception as e:
                print(f"[ERROR] Error saving embeddings: {e}")
    
    def add_documents(self, documents: List[Tuple[str, str, Dict]], embeddings: np.ndarray):
        """
        Add documents and their embeddings to the store
        
        Args:
            documents: List of tuples (file_path, text_chunk, metadata)
            embeddings: Array of embedding vectors
        """
        print(f"[DEBUG] Adding {len(documents)} documents to embedding store")
        
        if len(documents) == 0:
            print("[WARNING] No documents to add")
            return
        
        # Format documents for storage
        formatted_docs = []
        for i, (file_path, text_chunk, metadata) in enumerate(documents):
            formatted_docs.append({
                "id": f"doc_{int(time.time())}_{i}",
                "file_path": file_path,
                "text_chunk": text_chunk,
                "metadata": metadata or {}
            })
        
        # Add to existing documents and embeddings
        if self.embeddings is None:
            self.embeddings = embeddings
            self.documents = formatted_docs
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
            self.documents.extend(formatted_docs)
        
        # Save to disk
        self._save_embeddings()
        
        print(f"[DEBUG] Now have {len(self.documents)} documents in store")
    
    def similarity_search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of document dictionaries with similarity scores
        """
        print(f"[DEBUG] Performing similarity search with top_k={top_k}")
        
        if self.embeddings is None or len(self.documents) == 0:
            print("[WARNING] No documents in store")
            return []
        
        # Reshape query embedding to 2D array for sklearn
        query_embedding_2d = query_embedding.reshape(1, -1)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding_2d,