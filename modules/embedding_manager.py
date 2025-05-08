import numpy as np
from sentence_transformers import SentenceTransformer
import atexit
import traceback
import gc
import torch
from typing import List, Union, Optional


class EmbeddingManager:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', use_multiprocessing: bool = False, device: Optional[str] = None):
        """
        Initialize the embedding manager with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence transformer model to use
            use_multiprocessing: Whether to use multiprocessing for batch encoding
            device: Device to use for inference ('cpu', 'cuda', etc.). If None, will use CUDA if available.
        """
        print(f"[DEBUG] EmbeddingManager.__init__ - model_name: {model_name}, use_multiprocessing: {use_multiprocessing}")
        
        # Determine device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"[DEBUG] Using device: {device}")
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.use_multiprocessing = use_multiprocessing
        self.model_name = model_name
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        
        print(f"[DEBUG] Model loaded with embedding dimension: {self.embedding_dimension}")
        print("[DEBUG] EmbeddingManager initialized successfully")
        
        # Register cleanup on exit
        atexit.register(self.cleanup)
        print("[DEBUG] Cleanup registered with atexit")
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Size of batches for processing
            
        Returns:
            Numpy array of embeddings with shape (len(texts), embedding_dimension)
        """
        print(f"[DEBUG] generate_embeddings - Processing {len(texts)} texts")
        if not texts:
            print("[DEBUG] No texts provided, returning empty array")
            return np.array([])
        
        try:
            print("[DEBUG] Encoding texts with SentenceTransformer")
            # Use explicit batch size and disable multiprocessing if specified
            encode_params = {
                'convert_to_tensor': False,
                'batch_size': batch_size,
                'show_progress_bar': False,
                'convert_to_numpy': True,
                'normalize_embeddings': True,  # Normalize for cosine similarity
            }
            
            # Remove num_workers parameter as it's not supported
            if not self.use_multiprocessing:
                print("[DEBUG] Multiprocessing disabled")
                # Don't add num_workers parameter
            
            # Process in smaller batches if there are many texts
            if len(texts) > 1000:
                print("[DEBUG] Large number of texts, processing in chunks")
                all_embeddings = []
                chunk_size = 500  # Process 500 texts at a time
                
                for i in range(0, len(texts), chunk_size):
                    print(f"[DEBUG] Processing chunk {i//chunk_size + 1}/{(len(texts) + chunk_size - 1)//chunk_size}")
                    chunk_texts = texts[i:i+chunk_size]
                    chunk_embeddings = self.model.encode(chunk_texts, **encode_params)
                    all_embeddings.append(chunk_embeddings)
                    
                    # Force garbage collection
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Combine all chunks
                embeddings = np.vstack(all_embeddings)
                print(f"[DEBUG] Combined embeddings with shape {embeddings.shape}")
            else:
                # Process all at once for smaller batches
                print(f"[DEBUG] Encode parameters: {encode_params}")
                embeddings = self.model.encode(texts, **encode_params)
                print(f"[DEBUG] Embeddings generated with shape {embeddings.shape}")
            
            # Ensure 2D array
            if embeddings.ndim == 1:
                print("[DEBUG] Reshaping 1D embeddings to 2D")
                embeddings = embeddings.reshape(1, -1)
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return embeddings
        except Exception as e:
            print(f"[ERROR] Error in generate_embeddings: {e}")
            traceback.print_exc()
            raise
    
    def generate_query_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single query text.
        
        Args:
            text: Query text to embed
            
        Returns:
            Numpy array of embedding with shape (1, embedding_dimension)
        """
        print(f"[DEBUG] generate_query_embedding - Processing query: {text[:50]}...")
        if not text:
            print("[WARNING] Empty query text provided")
            # Return zero vector with correct dimension
            return np.zeros((1, self.embedding_dimension))
            
        try:
            print("[DEBUG] Encoding query with SentenceTransformer")
            embedding = self.model.encode(
                text, 
                convert_to_tensor=False,
                batch_size=1,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,  # Normalize for cosine similarity
                device=self.device
            )
            print(f"[DEBUG] Query embedding generated with shape {embedding.shape}")
            
            # Ensure 2D array
            result = embedding.reshape(1, -1)
            print(f"[DEBUG] Reshaped query embedding to {result.shape}")
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available() and self.device == 'cuda':
                torch.cuda.empty_cache()
                
            return result
        except Exception as e:
            print(f"[ERROR] Error in generate_query_embedding: {e}")
            traceback.print_exc()
            # Return zero vector with correct dimension as fallback
            return np.zeros((1, self.embedding_dimension))
    
    def batch_encode_queries(self, queries: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode multiple queries efficiently.
        
        Args:
            queries: List of query strings
            batch_size: Batch size for encoding
            
        Returns:
            Numpy array of embeddings with shape (len(queries), embedding_dimension)
        """
        print(f"[DEBUG] batch_encode_queries - Processing {len(queries)} queries")
        return self.generate_embeddings(queries, batch_size=batch_size)
    
    def cleanup(self):
        """Clean up resources used by the embedding manager"""
        print("[DEBUG] EmbeddingManager.cleanup called")
        try:
            # Clean up any multiprocessing resources
            if hasattr(self, 'pool'):
                print("[DEBUG] Closing and joining pool")
                self.pool.close()
                self.pool.join()
                
            # Clean up CUDA cache if using GPU
            if torch.cuda.is_available():
                print("[DEBUG] Clearing CUDA cache")
                torch.cuda.empty_cache()
                
            print("[DEBUG] Cleanup completed successfully")
        except Exception as e:
            print(f"[ERROR] Error in cleanup: {e}")
            traceback.print_exc()
    
    def __del__(self):
        """Ensure cleanup on deletion"""
        print("[DEBUG] EmbeddingManager.__del__ called")
        try:
            self.cleanup()
            # Unregister the cleanup
            try:
                print("[DEBUG] Unregistering cleanup from atexit")
                atexit.unregister(self.cleanup)
                print("[DEBUG] Cleanup unregistered successfully")
            except:
                print("[DEBUG] Failed to unregister cleanup (may already be unregistered)")
        except Exception as e:
            print(f"[ERROR] Error in __del__: {e}")
            traceback.print_exc()



