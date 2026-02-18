
import logging

def setup_logger(name: str, log_file: str = "devagent.log", level=logging.INFO):
    """Function to setup a logger; can be called from different modules"""
    
    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    
    # Ensure log directory exists if needed, for now just root folder as per plan
    # If the user wants specific folder, they can specify path in log_file
    
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding multiple handlers if logger is already configured
    if not logger.handlers:
        logger.addHandler(handler)

    return logger
