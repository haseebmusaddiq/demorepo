"""
Debug script for testing the LLMManager in isolation
"""
import logging
from logging_config import setup_logging
from modules.llm_manager import LLMManager

# Set up logging with DEBUG level
setup_logging(log_level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main():
    """Test the LLMManager with sample data"""
    logger.info("Starting LLMManager debug test")
    
    # Sample configuration
    config = {
        "model": "simple",
        "debug": True,
        "max_tokens": 100
    }
    
    # Initialize LLMManager
    llm_manager = LLMManager(config)
    logger.info("LLMManager initialized")
    
    # Sample documents
    sample_docs = [
        "Artificial intelligence (AI) is intelligence demonstrated by machines. "
        "Machine learning is a subset of AI that focuses on the development of algorithms "
        "that can access data and use it to learn for themselves.",
        
        "Natural Language Processing (NLP) is a field of AI that gives machines the ability "
        "to read, understand and derive meaning from human languages. "
        "It combines computational linguistics with statistical models.",
        
        "Deep learning is part of a broader family of machine learning methods based on "
        "artificial neural networks with representation learning. "
        "It can be supervised, semi-supervised or unsupervised."
    ]
    
    # Test queries
    test_queries = [
        "What is artificial intelligence?",
        "Explain machine learning",
        "How does NLP work?",
        "What is the relationship between AI and machine learning?",
        "Tell me about neural networks",
        "What is a topic not in the documents?"
    ]
    
    # Test each query
    for query in test_queries:
        logger.info(f"\n\n--- Testing query: '{query}' ---")
        
        # Generate response
        response = llm_manager.generate_response(query, sample_docs)
        
        # Log the result
        logger.info(f"Query: {query}")
        logger.info(f"Response: {response}")
        
        print(f"\nQuery: {query}")
        print(f"Response: {response}\n")
        print("-" * 80)

if __name__ == "__main__":
    main()
    logger.info("LLMManager debug test completed")