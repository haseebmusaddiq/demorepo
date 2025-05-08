# Import compatibility layer first
import huggingface_compat

from flask import Flask, request, jsonify, render_template
import yaml
import os
import sys
import traceback
import logging
from modules.vector_store import VectorStore
from modules.llm_manager import LLMManager
from modules.document_processor import DocumentProcessor
from modules.embedding_manager import EmbeddingManager
from sentence_transformers import CrossEncoder
from typing import List, Dict
import numpy as np
import gc
import re
import time
import subprocess

# Create logs directory if it doesn't exist
logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(logs_dir, 'app.log'), mode='a')
    ]
)

# Get logger for this module
logger = logging.getLogger(__name__)
logger.info("Starting RAG service application")


def create_components():
    """Create and initialize all components"""
    print("[DEBUG] Starting create_components()")
    try:
        # Load configuration
        config_path = "config/config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("[DEBUG] Config loaded successfully")
        
        # Initialize components with multiprocessing disabled
        print("[DEBUG] Initializing DocumentProcessor")
        doc_processor = DocumentProcessor()
        
        print("[DEBUG] Initializing EmbeddingManager")
        embedding_manager = EmbeddingManager(use_multiprocessing=False)
        
        print("[DEBUG] Initializing VectorStore")
        vector_store = VectorStore()
        
        print("[DEBUG] Initializing LLMManager")
        llm_manager = LLMManager(config_path)  # Pass the path instead of the config dict
        
        print("[DEBUG] Initializing CrossEncoder")
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        print("[DEBUG] All components initialized successfully")
        return config, doc_processor, embedding_manager, vector_store, llm_manager, cross_encoder
    except Exception as e:
        print(f"[ERROR] Error in create_components: {e}")
        traceback.print_exc()
        raise

def index_documents_on_startup(doc_processor, embedding_manager, vector_store):
    """Index documents during application startup"""
    print("[DEBUG] Starting document indexing on startup")
    try:
        # Load and process documents
        print("[DEBUG] Loading documents")
        documents = doc_processor.load_documents()
        print(f"[DEBUG] Loaded {len(documents) if documents else 0} documents")
        
        if not documents:
            print("[WARNING] No documents found to process during startup indexing")
            return False
        
        # Generate embeddings in batches without multiprocessing
        print("[DEBUG] Generating embeddings")
        texts = [doc[1] for doc in documents]
        
        # Process in smaller batches to avoid memory issues
        batch_size = 100  # Adjust based on your system's capabilities
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            print(f"[DEBUG] Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = embedding_manager.generate_embeddings(batch_texts)
            all_embeddings.append(batch_embeddings)
            print(f"[DEBUG] Batch {i//batch_size + 1} processed successfully")
        
        # Combine all batches
        if all_embeddings:
            embeddings = np.vstack(all_embeddings)
            print(f"[DEBUG] Combined embeddings with shape {embeddings.shape}")
        else:
            print("[WARNING] No embeddings generated during startup indexing")
            return False
        
        # Store in vector store
        print("[DEBUG] Storing embeddings in vector store")
        vector_store.upsert_embeddings(documents, embeddings)
        print("[DEBUG] Embeddings stored successfully")
        
        # Force garbage collection to clean up resources
        gc.collect()
        print("[DEBUG] Garbage collection performed")
        
        print("[DEBUG] Startup indexing completed successfully")
        return True
    except Exception as e:
        print(f"[ERROR] Error in startup indexing: {e}")
        traceback.print_exc()
        return False

def rerank_documents(query: str, documents: List[Dict]) -> List[Dict]:
    """Rerank documents based on additional relevance criteria"""
    print("[DEBUG] Starting rerank_documents")
    global cross_encoder
    
    # Small batch processing, no need for multiprocessing
    pairs = [[query, doc['text_chunk']] for doc in documents]
    print(f"[DEBUG] Created {len(pairs)} pairs for reranking")
    
    try:
        print("[DEBUG] Predicting with cross_encoder")
        scores = cross_encoder.predict(pairs)
        print(f"[DEBUG] Got {len(scores)} scores from cross_encoder")
        
        rerank_weight = config['retrieval'].get('rerank_weight', 0.7)
        print(f"[DEBUG] Using rerank_weight: {rerank_weight}")
        
        for doc, score in zip(documents, scores):
            doc['rerank_score'] = float(score)
            doc['final_score'] = (
                doc.get('combined_score', 0) * (1 - rerank_weight) +
                doc['rerank_score'] * rerank_weight
            )
        
        documents.sort(key=lambda x: x['final_score'], reverse=True)
        print("[DEBUG] Documents sorted by final_score")
        return documents
    except Exception as e:
        print(f"[ERROR] Error in rerank_documents: {e}")
        traceback.print_exc()
        return documents


# Initialize Flask app
app = Flask(__name__)

# Initialize components only in the main process
print("[DEBUG] Before component initialization check")
# Create a flag file to track if indexing has been done
indexing_flag_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp', 'indexing_done.flag')

# Make sure the temp directory exists
os.makedirs(os.path.dirname(indexing_flag_file), exist_ok=True)

# Only initialize components in the main process or the reloader process
if not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
    print("[DEBUG] Initializing components in main process")
    config, doc_processor, embedding_manager, vector_store, llm_manager, cross_encoder = create_components()
    print("[DEBUG] Components initialized successfully")
    
    # Check if indexing has already been done in this session
    should_index = True
    if app.debug:
        if os.path.exists(indexing_flag_file):
            # Check if the flag file is recent (less than 1 hour old)
            file_age = time.time() - os.path.getmtime(indexing_flag_file)
            if file_age < 3600:  # 1 hour in seconds
                should_index = False
                print("[DEBUG] Skipping indexing as it was recently done")
    
    # Index documents during startup if needed
    if should_index:
        print("[DEBUG] Starting document indexing during startup")
        indexing_success = index_documents_on_startup(doc_processor, embedding_manager, vector_store)
        print(f"[DEBUG] Startup indexing {'successful' if indexing_success else 'failed'}")
        
        # Create a flag file to indicate indexing has been done
        if indexing_success and app.debug:
            with open(indexing_flag_file, 'w') as f:
                f.write(f"Indexing completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")

@app.route('/')
def home():
    print("[DEBUG] Home route accessed")
    return render_template('index.html')

@app.route('/index_documents', methods=['POST'])
def index_documents():
    print("[DEBUG] index_documents route accessed")
    try:
        # Re-index documents on demand (this is now optional since we index on startup)
        print("[DEBUG] Re-indexing documents on demand")
        success = index_documents_on_startup(doc_processor, embedding_manager, vector_store)
        
        if success:
            print("[DEBUG] Re-indexing completed successfully")
            return jsonify({
                'status': 'success', 
                'message': 'Successfully re-indexed documents'
            })
        else:
            print("[DEBUG] Re-indexing failed")
            return jsonify({'status': 'error', 'message': 'Failed to re-index documents'})
    except Exception as e:
        print(f"[ERROR] Error in index_documents: {e}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/query', methods=['POST'])
def query():
    print("[DEBUG] query route accessed")
    try:
        user_query = request.json['query']
        print(f"[DEBUG] Processing query: {user_query[:50]}...")
        
        # Generate query embedding
        print("[DEBUG] Generating query embedding")
        query_embedding = embedding_manager.generate_query_embedding(user_query)
        print("[DEBUG] Query embedding generated")
        
        # Use hybrid search for better results
        print("[DEBUG] Performing hybrid search")
        similar_docs = vector_store.hybrid_search(
            query_embedding=query_embedding,
            query_text=user_query,
            top_k=config['retrieval']['top_k'],
            alpha=config['retrieval']['hybrid_search_alpha']
        )
        print(f"[DEBUG] Found {len(similar_docs)} similar documents")
        
        # Rerank results
        print("[DEBUG] Reranking documents")
        reranked_docs = rerank_documents(user_query, similar_docs)
        print("[DEBUG] Documents reranked")
        
        # Build enhanced prompt
        print("[DEBUG] Building prompt")
        relevant_texts = [
            f"From document '{doc['file_path']}':\n{doc['text_chunk']}"
            for doc in reranked_docs
        ]
        
        # Print the relevant texts for debugging
        print("\n=== RELEVANT TEXTS ===")
        for i, text in enumerate(relevant_texts[:3]):  # Show first 3 for brevity
            print(f"Text {i+1}:\n{text[:200]}...\n")
        print("=== END OF RELEVANT TEXTS ===\n")
        
        prompt = llm_manager.build_prompt(
            query=user_query,
            contexts=relevant_texts,
            max_context_length=config['llm'].get('max_context_length', 3000)
        )
        print("[DEBUG] Prompt built")
        
        # Generate response
        print("[DEBUG] Generating response")
        response = llm_manager.generate_response(prompt)
        print("[DEBUG] Response generated")
        
        print("[DEBUG] query completed successfully")
        return jsonify({
            'status': 'success',
            'response': response,
            'sources': [
                {
                    'file_path': doc['file_path'],
                    'similarity_score': doc.get('final_score', 0),
                    'excerpt': doc['text_chunk'][:200] + '...'
                }
                for doc in reranked_docs
            ]
        })
    except Exception as e:
        print(f"[ERROR] Error in query: {e}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)})
@app.route('/generate_answer', methods=['POST'])
def generate_answer():
    """Generate an answer based on a request, with optional negation"""
    print("[DEBUG] generate_answer route accessed")
    try:
        data = request.json
        if not data:
            return jsonify({'status': 'error', 'message': 'No data provided'}), 400
            
        user_query = data.get('query')
        negate = data.get('negate', False)  # Get the negate flag from the request
        
        if not user_query:
            return jsonify({'status': 'error', 'message': 'No query provided'}), 400
        
        print(f"[DEBUG] Processing query: {user_query[:50]}... (negate={negate})")
        
        try:
            # 1. Generate query embedding
            print("[DEBUG] Generating query embedding")
            query_embedding = embedding_manager.generate_query_embedding(user_query)
            print("[DEBUG] Query embedding generated successfully")
        except Exception as e:
            print(f"[ERROR] Error generating query embedding: {e}")
            traceback.print_exc()
            return jsonify({
                'status': 'error', 
                'message': f"Failed to generate query embedding: {str(e)}"
            }), 500
        
        try:
            # 2. Retrieve relevant documents
            print("[DEBUG] Retrieving relevant documents")
            similar_docs = vector_store.hybrid_search(
                query_embedding=query_embedding,
                query_text=user_query,
                top_k=config['retrieval']['top_k'],
                alpha=config['retrieval']['hybrid_search_alpha']
            )
            print(f"[DEBUG] Retrieved {len(similar_docs)} documents")
        except Exception as e:
            print(f"[ERROR] Error retrieving documents: {e}")
            traceback.print_exc()
            return jsonify({
                'status': 'error', 
                'message': f"Failed to retrieve relevant documents: {str(e)}"
            }), 500
        
        try:
            # 3. Rerank documents if configured
            print("[DEBUG] Reranking documents")
            if config['retrieval'].get('use_reranker', False):
                reranked_docs = reranker.rerank(
                    query=user_query,
                    documents=similar_docs,
                    top_k=config['retrieval'].get('reranker_top_k', 5)
                )
                print(f"[DEBUG] Reranked to {len(reranked_docs)} documents")
            else:
                reranked_docs = similar_docs[:config['retrieval'].get('reranker_top_k', 5)]
                print("[DEBUG] Skipped reranking, using top documents")
        except Exception as e:
            print(f"[ERROR] Error reranking documents: {e}")
            traceback.print_exc()
            # Fall back to using the similar docs without reranking
            reranked_docs = similar_docs[:config['retrieval'].get('reranker_top_k', 5)]
            print("[DEBUG] Using similar docs without reranking due to error")
        
        try:
            # 4. Extract relevant text from documents
            print("[DEBUG] Extracting relevant text")
            relevant_texts = [doc['text_chunk'] for doc in reranked_docs]
            print(f"[DEBUG] Extracted {len(relevant_texts)} text chunks")
        except Exception as e:
            print(f"[ERROR] Error extracting text: {e}")
            traceback.print_exc()
            return jsonify({
                'status': 'error', 
                'message': f"Failed to extract relevant text: {str(e)}"
            }), 500
        
        try:
            # 5. Build prompt and generate answer
            print("[DEBUG] Building prompt and generating answer")
            
            if negate:
                # Create a negated version of the query
                modified_query = f"Original question: {user_query}\nProvide a response that directly contradicts or negates the original question."
                
                # Build prompt with the negated query
                prompt = llm_manager.build_prompt(
                    query=modified_query,
                    contexts=relevant_texts,
                    max_context_length=config['llm'].get('max_context_length', 3000)
                )
                
                print("[DEBUG] Built prompt for contrary response")
            else:
                # Standard prompt building
                prompt = llm_manager.build_prompt(
                    query=user_query,
                    contexts=relevant_texts,
                    max_context_length=config['llm'].get('max_context_length', 3000)
                )
                
                print("[DEBUG] Built standard prompt")
        except Exception as e:
            print(f"[ERROR] Error building prompt: {e}")
            traceback.print_exc()
            return jsonify({
                'status': 'error', 
                'message': f"Failed to build prompt: {str(e)}"
            }), 500
        
        try:
            # Generate response
            response = llm_manager.generate_response(prompt)
            print(f"[DEBUG] negate = {negate}  Answer generated successfully \n{response[:100]}...")
        except Exception as e:
            print(f"[ERROR] Error generating response: {e}")
            traceback.print_exc()
            return jsonify({
                'status': 'error', 
                'message': f"Failed to generate response: {str(e)}"
            }), 500
        
        # Prepare the response
        result = {
            'status': 'success',
            'answer': response,
            'sources': [
                {
                    'file_path': doc.get('file_path', 'Unknown'),
                    'similarity_score': doc.get('final_score', 0),
                    'excerpt': doc.get('text_chunk', '')[:200] + '...'
                }
                for doc in reranked_docs[:5]
            ]
        }
        
        # Add negated_query field for contrary responses
        if negate:
            result['negated_query'] = "Providing an alternative perspective or challenging the assumptions in the original question."
        
        return jsonify(result)
        
    except Exception as e:
        print(f"[ERROR] Error in generate_answer: {e}")
        traceback.print_exc()
        return jsonify({
            'status': 'error', 
            'message': f"Failed to generate answer: {str(e)}"
        }), 500
@app.route('/index_status', methods=['GET'])
def index_status():
    """Check if documents are indexed"""
    try:
        # Check if the vector store has documents
        indexed = vector_store.is_initialized()
        
        return jsonify({
            'status': 'success',
            'indexed': indexed,
            'document_count': vector_store.get_document_count() if indexed else 0
        })
    except Exception as e:
        print(f"[ERROR] Error in index_status: {e}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f"Failed to check index status: {str(e)}"
        }), 500
if __name__== '__main__':
    # Run the app without debug mode to avoid WERKZEUG_SERVER_FD error
    print("[INFO] Starting Flask application")
    try:
        # First try with debug mode disabled
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False, use_debugger=False)
    except Exception as e:
        print(f"[ERROR] Failed to start Flask app: {e}")
        # If that fails, try with a different configuration
        try:
            from werkzeug.serving import run_simple
            print("[INFO] Attempting to start with werkzeug.serving.run_simple")
            run_simple('0.0.0.0', 5000, app, use_debugger=False, use_reloader=False)
        except Exception as e2:
            print(f"[ERROR] Failed to start with run_simple: {e2}")
            print("[INFO] Please try running with: flask --app app run --host=0.0.0.0 --port=5000")




