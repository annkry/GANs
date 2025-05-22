import logging

def setup_logging():
    """Initializes logging to the console."""
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')