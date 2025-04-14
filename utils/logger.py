import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def print_flush(*args, **kwargs):
    """
    Custom function that sends messages to the Python logging system at INFO level
    instead of printing directly to stdout. Ultimately, these messages will be
    funneled to Loguru through the InterceptHandler.
    """
    message = " ".join(str(arg) for arg in args)
    logger.info(message)