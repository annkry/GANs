"""
    Provides logging setup functionality to standardize log outputs.
"""

import logging

def setup_logging():
    """Initializes logging configuration for the project."""
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')