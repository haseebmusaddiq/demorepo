import numpy as np
import faiss
from typing import List, Dict
import yaml
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import gc
import os
import traceback  # Add this import for error stack traces

class VectorStore:
    def __init__(self, config_path: str = "config/config.yaml"):
        print("[DEBUG] Initializing VectorStore")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.documents = []
        self.index = None
        self.dimension = None
        self.vectorizer = None
        print("[DEBUG] VectorStore initialized")

    def _cleanup(self):
        """Cleanup resources"""
        print("[DEBUG] VectorStore._cleanup called")
        if self.index is not None:
            print("[DEBUG] Resetting FAISS index")
            self.index.reset()
            self.index = None
        self.clear()
        gc.collect()
        print("[DEBUG] VectorStore cleanup completed")

    def clear(self):
        """Clear all documents and reset the index"""
        print("[DEBUG] VectorStore.clear called")
        self.documents = []
        if self.index is not None:
            print("[DEBUG] Resetting FAISS index in clear()")
            self.index.reset()
            self.index = None
        gc.collect()
        print("[DEBUG] VectorStore clear completed")

    def initialize_index(self, dimension: int):
        """Initialize FAISS index with IVF for better search performance"""
        print(f"[DEBUG] Initializing FAISS index with dimension {dimension}")
        self.dimension = dimension
        
        # Use a simpler index for smaller datasets to avoid segfaults
        if len(self.documents) < 1000:
            print("[DEBUG] Using simple FlatIP index due to small dataset size")
            self.index = faiss.IndexFlatIP(dimension)
            return
            
        # Use IVFFlat index for better performance with cosine similarity
        try:
            nlist = max(4, min(len(self.documents) // 10, 100))  # number of clusters, capped
            print(f"[DEBUG] Creating IVFFlat index with {nlist} clusters")
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            
            # Need to train with some data
            if len(self.documents) > 0:
                print("[DEBUG] Training index with existing documents")
                embeddings = np.stack([doc['embedding'] for doc in self.documents])
                self.index.train(embeddings)
                print("[DEBUG] Index trained successfully")
        except Exception as e:
            print(f"[ERROR] Error initializing IVFFlat index: {e}")
            print("[DEBUG] Falling back to simple FlatIP index")
            self.index = faiss.IndexFlatIP(dimension)

    def add_documents(self, documents: List[Dict], embeddings: np.ndarray):
        """Add documents and their embeddings to the store"""
        print(f"[DEBUG] Adding {len(documents)} documents to vector store")
        
        # Safety check
        if len(documents) == 0:
            print("[WARNING] No documents to add")
            return
            
        # Normalize embeddings for cosine similarity
        print("[DEBUG] Normalizing embeddings")
        embeddings = normalize(embeddings, axis=1)
        
        # Initialize index if needed
        if self.index is None:
            print("[DEBUG] Index not initialized, initializing now")
            self.initialize_index(embeddings.shape[1])
        
        # Add to documents list
        print("[DEBUG] Adding documents to internal list")
        for doc, emb in zip(documents, embeddings):
            self.documents.append({
                'file_path': doc[0],
                'text_chunk': doc[1],
                'embedding': emb,
                'metadata': doc[2] if len(doc) > 2 else {}
            })
        
        # Add to FAISS index
        try:
            print("[DEBUG] Adding embeddings to FAISS index")
            if not hasattr(self.index, 'is_trained') or self.index.is_trained:
                print("[DEBUG] Index is already trained or doesn't require training")
                self.index.add(embeddings)
            else:
                print("[DEBUG] Training index before adding embeddings")
                self.index.train(embeddings)
                self.index.add(embeddings)
            print("[DEBUG] Embeddings added to index successfully")
        except Exception as e:
            print(f"[ERROR] Error adding embeddings to index: {e}")
            # Try to recover by rebuilding the index
            print("[DEBUG] Attempting to rebuild index")
            self._rebuild_index()

    def upsert_embeddings(self, documents: List[Dict], embeddings: np.ndarray):
        """Update or insert new embeddings into the store"""
        print(f"[DEBUG] Upserting {len(documents)} documents")
        
        # Safety check
        if len(documents) == 0:
            print("[WARNING] No documents to upsert")
            return
            
        # If index is empty, just add the documents
        if self.index is None or len(self.documents) == 0:
            print("[DEBUG] No existing index, adding documents directly")
            self.add_documents(documents, embeddings)
            return

        # For safety, let's rebuild the index from scratch
        print("[DEBUG] Rebuilding index for safety")
        
        # First, update our document list
        updated_docs = []
        for doc, emb in zip(documents, embeddings):
            file_path = doc[0]
            text_chunk = doc[1]
            metadata = doc[2] if len(doc) > 2 else {}
            
            # Check if document already exists
            existing_idx = None
            for idx, existing_doc in enumerate(self.documents):
                if existing_doc['file_path'] == file_path and existing_doc['text_chunk'] == text_chunk:
                    existing_idx = idx
                    break
            
            if existing_idx is not None:
                # Update existing document
                self.documents[existing_idx]['embedding'] = emb
                self.documents[existing_idx]['metadata'] = metadata
            else:
                # Add new document
                updated_docs.append({
                    'file_path': file_path,
                    'text_chunk': text_chunk,
                    'embedding': emb,
                    'metadata': metadata
                })
        
        # Add new documents to our list
        self.documents.extend(updated_docs)
        
        # Rebuild the index
        self._rebuild_index()
        print("[DEBUG] Index rebuilt successfully")

    def _rebuild_index(self):
        """Rebuild the FAISS index from scratch"""
        print("[DEBUG] Rebuilding FAISS index from scratch")
        if not self.documents:
            print("[WARNING] No documents to build index with")
            return
            
        try:
            # Clean up old index
            if self.index is not None:
                print("[DEBUG] Resetting old index")
                self.index.reset()
                self.index = None
                
            # Force garbage collection
            gc.collect()
            
            # Stack embeddings
            print("[DEBUG] Stacking embeddings")
            embeddings = np.stack([doc['embedding'] for doc in self.documents])
            dimension = embeddings.shape[1]
            
            # Use a simpler index for smaller datasets
            if len(self.documents) < 1000:
                print("[DEBUG] Using simple FlatIP index due to small dataset size")
                new_index = faiss.IndexFlatIP(dimension)
                new_index.add(embeddings)
            else:
                # Create new index
                nlist = max(4, min(len(self.documents) // 10, 100))  # number of clusters, capped
                print(f"[DEBUG] Creating new IVFFlat index with {nlist} clusters")
                quantizer = faiss.IndexFlatIP(dimension)
                new_index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
                
                # Train and add all embeddings
                print("[DEBUG] Training new index")
                new_index.train(embeddings)
                print("[DEBUG] Adding embeddings to new index")
                new_index.add(embeddings)
            
            # Replace old index
            self.index = new_index
            print("[DEBUG] Index rebuilt successfully")
            
            # Force garbage collection again
            gc.collect()
        except Exception as e:
            print(f"[ERROR] Error rebuilding index: {e}")
            # Fall back to a simple index
            print("[DEBUG] Falling back to simple FlatIP index")
            try:
                dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dimension)
                self.index.add(embeddings)
                print("[DEBUG] Fallback index created successfully")
            except Exception as e2:
                print(f"[ERROR] Error creating fallback index: {e2}")
                # Last resort - empty index
                print("[WARNING] Creating empty index as last resort")
                if self.dimension is not None:
                    self.index = faiss.IndexFlatIP(self.dimension)
                else:
                    print("[ERROR] Cannot create empty index without dimension")

    def __del__(self):
        """Destructor to ensure cleanup"""
        print("[DEBUG] VectorStore.__del__ called")
        self._cleanup()

    def batch_process_documents(self, documents: List[Dict], batch_size: int = 100):
        """Process documents in batches to manage memory usage"""
        print(f"[DEBUG] batch_process_documents called with {len(documents)} documents")
        try:
            total_batches = (len(documents) + batch_size - 1) // batch_size
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i
 + batch_size]
                print(f"[DEBUG] Processing batch {i//batch_size + 1}/{total_batches} with {len(batch)} documents")
                
                # Process batch directly
                results = []
                for doc in batch:
                    try:
                        processed_doc = self._process_document(doc)
                        results.append(processed_doc)
                    except Exception as e:
                        print(f"[ERROR] Error processing document: {e}")
                        # Continue with other documents
                        continue
                
                if not results:
                    print("[WARNING] No documents processed in this batch")
                    continue
                    
                # Extract embeddings
                try:
                    print(f"[DEBUG] Extracting embeddings from {len(results)} processed documents")
                    embeddings = np.array([r['embedding'] for r in results if 'embedding' in r])
                    
                    if len(embeddings) == 0:
                        print("[WARNING] No valid embeddings found in processed documents")
                        continue
                        
                    print(f"[DEBUG] Adding {len(embeddings)} embeddings to vector store")
                    self.add_documents(batch[:len(embeddings)], embeddings)
                    print(f"[DEBUG] Batch {i//batch_size + 1} processed successfully")
                except Exception as e:
                    print(f"[ERROR] Error adding embeddings from batch: {e}")
                    traceback.print_exc()
                
                # Force garbage collection after each batch
                gc.collect()
        except Exception as e:
            print(f"[ERROR] Error in batch_process_documents: {e}")
            traceback.print_exc()
        finally:
            print("[DEBUG] Completed batch_process_documents, running garbage collection")
            gc.collect()

    def _process_document(self, document: Dict) -> Dict:
        """Process a single document"""
        print(f"[DEBUG] Processing document: {document.get('file_path', 'unknown')[:50]}...")
        try:
            # Check if document already has an embedding
            if 'embedding' in document:
                print("[DEBUG] Document already has embedding, using existing")
                return document
                
            # If document doesn't have an embedding, we need to generate one
            # This is a placeholder - in a real implementation, you would:
            # 1. Extract text from the document
            # 2. Generate an embedding using your embedding model
            # 3. Return the document with the embedding added
            
            # For now, we'll just return the document as is and log a warning
            print("[WARNING] Document processing not implemented, returning document without changes")
            return document
        except Exception as e:
            print(f"[ERROR] Error in _process_document: {e}")
            # Return the original document to avoid breaking the pipeline
            return document
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Search for similar documents using vector similarity"""
        print(f"[DEBUG] Searching for top {top_k} similar documents")
        
        if self.index is None:
            print("[WARNING] No index available for search")
            return []
            
        if len(self.documents) == 0:
            print("[WARNING] No documents in the store")
            return []
        
        try:
            # Normalize query embedding for cosine similarity
            query_embedding = normalize(query_embedding.reshape(1, -1), axis=1).astype(np.float32)
            
            # Search the index
            print("[DEBUG] Searching FAISS index")
            distances, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
            
            # Get the documents
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < 0 or idx >= len(self.documents):
                    print(f"[WARNING] Invalid index {idx} returned by FAISS")
                    continue
                    
                doc = self.documents[idx]
                results.append({
                    'file_path': doc['file_path'],
                    'text_chunk': doc['text_chunk'],
                    'metadata': doc['metadata'],
                    'similarity_score': float(distances[0][i])
                })
            
            print(f"[DEBUG] Found {len(results)} similar documents")
            return results
        except Exception as e:
            print(f"[ERROR] Error in search: {e}")
            traceback.print_exc()
            return []

    def hybrid_search(self, query_embedding: np.ndarray, query_text: str, top_k: int = 5, alpha: float = 0.5) -> List[Dict]:
        """Combine vector search with text-based search for better results"""
        print(f"[DEBUG] Performing hybrid search with alpha={alpha}")
        
        try:
            # Vector search
            vector_results = self.search(query_embedding, top_k=top_k*2)  # Get more results for reranking
            
            # Text search (if we have documents)
            if len(self.documents) > 0 and query_text:
                print("[DEBUG] Performing text search")
                try:
                    # Initialize TF-IDF vectorizer if needed
                    if self.vectorizer is None:
                        print("[DEBUG] Initializing TF-IDF vectorizer")
                        self.vectorizer = TfidfVectorizer()
                        corpus = [doc['text_chunk'] for doc in self.documents]
                        self.vectorizer.fit(corpus)
                    
                    # Transform query
                    query_vec = self.vectorizer.transform([query_text])
                    
                    # Transform documents
                    doc_vecs = self.vectorizer.transform([doc['text_chunk'] for doc in self.documents])
                    
                    # Calculate similarities
                    similarities = (doc_vecs @ query_vec.T).toarray().flatten()
                    
                    # Get top results
                    top_indices = similarities.argsort()[-top_k*2:][::-1]
                    
                    # Create text search results
                    text_results = []
                    for idx in top_indices:
                        if similarities[idx] > 0:  # Only include if there's some similarity
                            doc = self.documents[idx]
                            text_results.append({
                                'file_path': doc['file_path'],
                                'text_chunk': doc['text_chunk'],
                                'metadata': doc['metadata'],
                                'text_score': float(similarities[idx])
                            })
                except Exception as e:
                    print(f"[ERROR] Error in text search: {e}")
                    text_results = []
            else:
                print("[DEBUG] Skipping text search (no documents or query)")
                text_results = []
            
            # Combine results
            print("[DEBUG] Combining vector and text search results")
            combined_results = {}
            
            # Add vector results
            for doc in vector_results:
                doc_key = f"{doc['file_path']}:{doc['text_chunk'][:100]}"
                combined_results[doc_key] = {
                    'file_path': doc['file_path'],
                    'text_chunk': doc['text_chunk'],
                    'metadata': doc['metadata'],
                    'vector_score': doc['similarity_score'],
                    'text_score': 0.0,
                    'combined_score': alpha * doc['similarity_score']
                }
            
            # Add/update with text results
            for doc in text_results:
                doc_key = f"{doc['file_path']}:{doc['text_chunk'][:100]}"
                if doc_key in combined_results:
                    combined_results[doc_key]['text_score'] = doc['text_score']
                    combined_results[doc_key]['combined_score'] += (1 - alpha) * doc['text_score']
                else:
                    combined_results[doc_key] = {
                        'file_path': doc['file_path'],
                        'text_chunk': doc['text_chunk'],
                        'metadata': doc['metadata'],
                        'vector_score': 0.0,
                        'text_score': doc['text_score'],
                        'combined_score': (1 - alpha) * doc['text_score']
                    }
            
            # Sort by combined score
            results = list(combined_results.values())
            results.sort(key=lambda x: x['combined_score'], reverse=True)
            
            # Return top-k
            print(f"[DEBUG] Returning top {min(top_k, len(results))} results")
            return results[:top_k]
        except Exception as e:
            print(f"[ERROR] Error in hybrid_search: {e}")
            traceback.print_exc()
            # Fall back to vector search only
            print("[DEBUG] Falling back to vector search only")
            try:
                return self.search(query_embedding, top_k)
            except Exception as e2:
                print(f"[ERROR] Error in fallback search: {e2}")
                return []

    def is_initialized(self):
        """Check if the vector store has been initialized with documents"""
        print("[DEBUG] Checking if vector store is initialized")
        # Check if we have documents and an index
        is_init = self.index is not None and len(self.documents) > 0
        print(f"[DEBUG] Vector store initialized: {is_init}")
        return is_init

    def get_document_count(self):
        """Get the number of documents in the store"""
        print("[DEBUG] Getting document count")
        count = len(self.documents) if self.documents else 0
        print(f"[DEBUG] Document count: {count}")
        return count



