import os
import logging
import yaml
from pathlib import Path

def setup_logging(config_path: str = None) -> logging.Logger:
    """
    Set up logging configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml file
        
    Returns:
        Configured logger instance
    """
    # Load config
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create logs directory if it doesn't exist
    log_dir = config['paths']['logs_dir']
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=config['logging']['level'],
        format=config['logging']['format'],
        handlers=[
            logging.FileHandler(config['logging']['file']),
            logging.StreamHandler()
        ]
    )
    
    # Create logger
    logger = logging.getLogger('signal_extractor')
    return logger 