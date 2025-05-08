import logging
import os
import sys

def setup_logging(log_level=logging.INFO, log_file=None):
    """
    Set up logging configuration for the application
    
    Args:
        log_level: The logging level (default: INFO)
        log_file: Path to log file (default: None, logs to console only)
    """
    # Create logs directory if it doesn't exist and log_file is specified
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if log_file is specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    # Set LLMManager logger to DEBUG level
    logging.getLogger('rag_service.modules.llm_manager').setLevel(logging.DEBUG)
    
    # You can add more specific logger configurations here
    # For example:
    # logging.getLogger('werkzeug').setLevel(logging.WARNING)  # Reduce Flask logs
    
    logging.info(f"Logging initialized with level: {logging.getLevelName(log_level)}")
    if log_file:
        logging.info(f"Log file: {log_file}")

def enable_debug_mode():
    """
    Enable debug mode for all loggers
    """
    # Set root logger to DEBUG
    logging.getLogger().setLevel(logging.DEBUG)
    
    # Set all existing loggers to DEBUG
    for logger_name in logging.root.manager.loggerDict:
        logging.getLogger(logger_name).setLevel(logging.DEBUG)
    
    logging.info("Debug mode enabled for all loggers")