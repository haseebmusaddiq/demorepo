import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
from dotenv import load_dotenv
import vecs
import time
import requests
import base64
import traceback

class SupabaseVectorStore:
    """
    Vector store implementation using Supabase.
    Also handles file storage and retrieval.
    """
    
    def __init__(self, config=None, collection_name="documents"):
        """
        Initialize the Supabase Vector Store
        
        Args:
            config: Configuration dictionary
            collection_name: Name of the collection to use
        """
        print("[DEBUG] Initializing SupabaseVectorStore")
        self.config = config or {}
        self.collection_name = collection_name
        
        # Load environment variables from .env file if it exists
        load_dotenv()
        
        # Get Supabase credentials from environment variables or config
        self.supabase_url = os.getenv("SUPABASE_URL") or self.config.get("supabase", {}).get("url")
        self.supabase_key = os.getenv("SUPABASE_KEY") or self.config.get("supabase", {}).get("key")
        self.supabase_bucket = os.getenv("SUPABASE_BUCKET") or self.config.get("supabase", {}).get("bucket", "documents")
        
        # Check if we have valid credentials
        if not self.supabase_url or not self.supabase_key:
            print("[WARNING] Supabase credentials not found, using demo mode")
            self.demo_mode = True
            self.in_memory_docs = []
            self.in_memory_files = {}
        else:
            try:
                # Initialize Supabase client
                print("[DEBUG] Connecting to Supabase")
                self.demo_mode = False
                
                # Create storage bucket if it doesn't exist
                self._create_bucket_if_not_exists()
                print(f"[DEBUG] Connected to Supabase storage bucket: {self.supabase_bucket}")
            except Exception as e:
                print(f"[ERROR] Failed to connect to Supabase: {e}")
                self.demo_mode = True
                self.in_memory_docs = []
                self.in_memory_files = {}
    
    def _create_bucket_if_not_exists(self):
        """Create the storage bucket if it doesn't exist"""
        try:
            # Check if bucket exists
            headers = {
                "apikey": self.supabase_key,
                "Authorization": f"Bearer {self.supabase_key}"
            }
            response = requests.get(
                f"{self.supabase_url}/storage/buckets",
                headers=headers
            )
            
            # If bucket doesn't exist, create it
            buckets = response.json()
            bucket_exists = any(bucket.get("name") == self.supabase_bucket for bucket in buckets)
            
            if not bucket_exists:
                print(f"[DEBUG] Creating storage bucket: {self.supabase_bucket}")
                response = requests.post(
                    f"{self.supabase_url}/storage/buckets",
                    headers=headers,
                    json={
                        "name": self.supabase_bucket,
                        "public": False,
                        "file_size_limit": 52428800  # 50MB limit
                    }
                )
                if response.status_code not in (200, 201):
                    print(f"[ERROR] Failed to create bucket: {response.text}")
        except Exception as e:
            print(f"[ERROR] Error checking/creating bucket: {e}")
    
    def upload_file(self, file_path: str, file_content=None):
        """
        Upload a file to Supabase Storage
        
        Args:
            file_path: Path to the file or desired path in storage
            file_content: File content as bytes (optional, will read from file_path if not provided)
            
        Returns:
            URL of the uploaded file or None if failed
        """
        print(f"[DEBUG] Uploading file: {file_path}")
        
        if self.demo_mode:
            # In demo mode, just store in memory
            filename = os.path.basename(file_path)
            if file_content is None and os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    file_content = f.read()
            
            self.in_memory_files[filename] = {
                'content': file_content,
                'path': file_path,
                'uploaded_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            print(f"[DEBUG] File stored in memory: {filename}")
            return f"memory://{filename}"
        
        try:
            # Get file content if not provided
            if file_content is None and os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    file_content = f.read()
            
            if file_content is None:
                print(f"[ERROR] No content provided for file: {file_path}")
                return None
            
            # Upload to Supabase Storage
            filename = os.path.basename(file_path)
            headers = {
                "apikey": self.supabase_key,
                "Authorization": f"Bearer {self.supabase_key}",
                "Content-Type": "application/octet-stream"
            }
            
            response = requests.post(
                f"{self.supabase_url}/storage/v1/object/{self.supabase_bucket}/{filename}",
                headers=headers,
                data=file_content
            )
            
            if response.status_code in (200, 201):
                print(f"[DEBUG] File uploaded successfully: {filename}")
                # Get public URL
                file_url = f"{self.supabase_url}/storage/v1/object/public/{self.supabase_bucket}/{filename}"
                return file_url
            else:
                print(f"[ERROR] Failed to upload file: {response.text}")
                return None
        except Exception as e:
            print(f"[ERROR] Error uploading file: {e}")
            return None
    
    def get_file(self, filename: str):
        """
        Get a file from Supabase Storage
        
        Args:
            filename: Name of the file to retrieve
            
        Returns:
            File content as bytes or None if failed
        """
        print(f"[DEBUG] Getting file: {filename}")
        
        if self.demo_mode:
            # In demo mode, retrieve from memory
            if filename in self.in_memory_files:
                return self.in_memory_files[filename]['content']
            return None
        
        try:
            # Get from Supabase Storage
            headers = {
                "apikey": self.supabase_key,
                "Authorization": f"Bearer {self.supabase_key}"
            }
            
            response = requests.get(
                f"{self.supabase_url}/storage/v1/object/{self.supabase_bucket}/{filename}",
                headers=headers
            )
            
            if response.status_code == 200:
                print(f"[DEBUG] File retrieved successfully: {filename}")
                return response.content
            else:
                print(f"[ERROR] Failed to get file: {response.text}")
                return None
        except Exception as e:
            print(f"[ERROR] Error getting file: {e}")
            return None
    
    def list_files(self):
        """
        List all files in the storage bucket
        
        Returns:
            List of file information dictionaries
        """
        print("[DEBUG] Listing files")
        
        if self.demo_mode:
            # In demo mode, list from memory
            return [
                {
                    'name': filename,
                    'path': self.in_memory_files[filename]['path'],
                    'uploaded_at': self.in_memory_files[filename]['uploaded_at'],
                    'size': len(self.in_memory_files[filename]['content'])
                }
                for filename in self.in_memory_files
            ]
        
        try:
            # List from Supabase Storage
            headers = {
                "apikey": self.supabase_key,
                "Authorization": f"Bearer {self.supabase_key}"
            }
            
            response = requests.get(
                f"{self.supabase_url}/storage/v1/object/list/{self.supabase_bucket}",
                headers=headers
            )
            
            if response.status_code == 200:
                files = response.json()
                print(f"[DEBUG] Listed {len(files)} files")
                return files
            else:
                print(f"[ERROR] Failed to list files: {response.text}")
                return []
        except Exception as e:
            print(f"[ERROR] Error listing files: {e}")
            return []
    
    def add_documents(self, documents, embeddings):
        """
        Add documents and their embeddings to the vector store
        
        Args:
            documents: List of document tuples (file_path, text_chunk, metadata) or dictionaries
            embeddings: List of embedding vectors corresponding to the documents
        
        Returns:
            List of document IDs
        """
        print(f"[DEBUG] Adding {len(documents)} documents to vector store")
        
        if self.demo_mode:
            # In demo mode, just store in memory
            doc_ids = [f"doc_{i+len(self.in_memory_docs)}" for i in range(len(documents))]
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                # Handle tuple format (which is what we're getting from app.py)
                if isinstance(doc, tuple):
                    file_path = doc[0]
                    content = doc[1]
                    metadata = doc[2] if len(doc) > 2 else {}
                else:
                    # Handle dict format
                    content = doc.get("content", "")
                    metadata = doc.get("metadata", {})
                    file_path = metadata.get("file_path", "unknown")
                
                self.in_memory_docs.append({
                    "id": doc_ids[i],
                    "content": content,
                    "metadata": metadata,
                    "embedding": embedding
                })
            print(f"[DEBUG] Added {len(documents)} documents in demo mode")
            return doc_ids
        
        try:
            # Initialize vecs client if not already done
            if not hasattr(self, 'collection'):
                print("[DEBUG] Initializing vecs client")
                db_connection = f"postgresql://{os.getenv('SUPABASE_DB_USER')}:{os.getenv('SUPABASE_DB_PASSWORD')}@{os.getenv('SUPABASE_DB_HOST')}:{os.getenv('SUPABASE_DB_PORT')}/{os.getenv('SUPABASE_DB_NAME')}"
                vx = vecs.create_client(db_connection)
                self.collection = vx.get_or_create_collection(
                    name=self.collection_name,
                    dimension=len(embeddings[0]) if embeddings and len(embeddings) > 0 else 384
                )
                print(f"[DEBUG] Connected to collection: {self.collection_name}")
            
            # Prepare records for upsert
            records = []
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                # Handle tuple format (which is what we're getting from app.py)
                if isinstance(doc, tuple):
                    file_path = doc[0]
                    content = doc[1]
                    metadata = doc[2] if len(doc) > 2 else {}
                    doc_id = f"doc_{i}"
                else:
                    # Handle dict format
                    content = doc.get("content", "")
                    metadata = doc.get("metadata", {})
                    file_path = metadata.get("file_path", "unknown")
                    doc_id = metadata.get("id", f"doc_{i}")
                
                # Ensure metadata is JSON serializable
                if isinstance(metadata, dict):
                    try:
                        metadata_json = json.dumps(metadata)
                    except:
                        metadata_json = json.dumps({"error": "Metadata not serializable"})
                else:
                    metadata_json = json.dumps({"raw_metadata": str(metadata)})
                
                records.append((
                    doc_id,
                    embedding,
                    {
                        "content": content,
                        "metadata": metadata_json,
                        "file_path": file_path
                    }
                ))
            
            # Upsert records to collection
            self.collection.upsert(records)
            print(f"[DEBUG] Upserted {len(records)} records to collection")
            
            # Create index if needed
            if len(documents) > 100:  # Only create index when we have enough documents
                try:
                    self.collection.create_index()
                    print("[DEBUG] Created or updated vector index")
                except Exception as e:
                    print(f"[WARNING] Error creating index: {e}")
            
            return [record[0] for record in records]
        except Exception as e:
            print(f"[ERROR] Error adding documents to Supabase: {e}")
            traceback.print_exc()  # Add traceback for better debugging
            return []
    
    def similarity_search(
        self, 
        query_embedding: List[float], 
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query embedding
        
        Args:
            query_embedding: The embedding vector of the query
            top_k: Number of results to return
            filter_dict: Optional metadata filter
            
        Returns:
            List of documents with their similarity scores
        """
        print(f"[DEBUG] Searching for top {top_k} documents similar to query")
        
        if self.demo_mode:
            # In demo mode, compute similarities in memory
            if not self.in_memory_docs:
                return []
            
            results = []
            for doc in self.in_memory_docs:
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, doc["embedding"]) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc["embedding"])
                )
                
                # Apply filter if provided
                if filter_dict and not self._matches_filter(doc["metadata"], filter_dict):
                    continue
                
                results.append({
                    "id": doc["id"],
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "score": float(similarity)
                })
            
            # Sort by similarity score and take top_k
            results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
            print(f"[DEBUG] Found {len(results)} documents in demo mode")
            return results
        
        try:
            # Convert filter_dict to metadata filter if provided
            metadata_filter = None
            if filter_dict:
                metadata_filter = json.dumps(filter_dict)
            
            # Query the collection
            results = self.collection.query(
                data=query_embedding,
                limit=top_k,
                filters={"metadata": metadata_filter} if metadata_filter else None,
                include_metadata=True
            )
            
            # Format results
            formatted_results = []
            for result in results:
                doc_id, score, metadata = result
                
                # Parse metadata
                doc_metadata = json.loads(metadata.get("metadata", "{}"))
                content = metadata.get("content", "")
                
                formatted_results.append({
                    "id": doc_id,
                    "content": content,
                    "metadata": doc_metadata,
                    "score": float(score)
                })
            
            print(f"[DEBUG] Found {len(formatted_results)} documents")
            return formatted_results
        except Exception as e:
            print(f"[ERROR] Error searching Supabase: {e}")
            return []
    
    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Helper method to check if metadata matches filter criteria"""
        for key, value in filter_dict.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True

    def upsert_embeddings(self, documents, embeddings):
        """
        Update or insert new document embeddings into the store
        
        Args:
            documents: List of tuples (file_path, text_chunk, metadata) or dictionaries
            embeddings: Array of embedding vectors
        """
        print(f"[DEBUG] Upserting {len(documents)} documents to Supabase")
        
        # Convert numpy array to list if needed
        embeddings_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
        
        # Add documents to the store
        return self.add_documents(documents, embeddings_list)

    def hybrid_search(
        self, 
        query_embedding: List[float], 
        query_text: str, 
        top_k: int = 5, 
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector similarity and keyword matching
        
        Args:
            query_embedding: The embedding vector of the query
            query_text: The text of the query for keyword matching
            top_k: Number of results to return
            alpha: Weight for combining vector and keyword scores (1.0 = vector only)
        
        Returns:
            List of documents with their combined scores
        """
        print(f"[DEBUG] Performing hybrid search for query: {query_text[:50]}...")
        
        # Get vector search results
        vector_results = self.similarity_search(query_embedding, top_k=top_k*2)  # Get more results for reranking
        
        if self.demo_mode:
            # In demo mode, we'll do a simple keyword matching
            for result in vector_results:
                # Calculate simple keyword match score
                content = result["content"].lower()
                query_terms = query_text.lower().split()
                
                # Count matching terms
                term_matches = sum(1 for term in query_terms if term in content)
                keyword_score = term_matches / len(query_terms) if query_terms else 0
                
                # Combine scores
                result["keyword_score"] = keyword_score
                result["combined_score"] = alpha * result["score"] + (1 - alpha) * keyword_score
                
                # Add formatted fields for consistency with the app
                result["file_path"] = result["metadata"].get("file_path", "unknown")
                result["text_chunk"] = result["content"]
                result["similarity_score"] = result["score"]
        else:
            try:
                # In real mode with Supabase, we could use full-text search capabilities
                # This would require setting up proper text search indexes in Supabase
                
                # For now, we'll just add the same fields as in demo mode
                for result in vector_results:
                    # Add keyword score (placeholder for real implementation)
                    result["keyword_score"] = 0.5  # Placeholder
                    result["combined_score"] = alpha * result["score"]
                    
                    # Add formatted fields for consistency with the app
                    result["file_path"] = result["metadata"].get("file_path", "unknown")
                    result["text_chunk"] = result["content"]
                    result["similarity_score"] = result["score"]
            except Exception as e:
                print(f"[ERROR] Error in hybrid search: {e}")
        
        # Sort by combined score
        results = sorted(vector_results, key=lambda x: x.get("combined_score", 0), reverse=True)[:top_k]
        print(f"[DEBUG] Hybrid search returned {len(results)} results")
        return results










