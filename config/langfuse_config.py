from langfuse.langchain import CallbackHandler
from config.logger import setup_logger
import os
from dotenv import load_dotenv

load_dotenv()

logger = setup_logger(__name__)

# Validate Langfuse credentials
langfuse_secret = os.getenv("LANGFUSE_SECRET_KEY")
langfuse_public = os.getenv("LANGFUSE_PUBLIC_KEY")
langfuse_host = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

if not langfuse_secret or not langfuse_public:
    logger.warning(
        "Langfuse credentials not found. Traces will NOT be sent. "
        "Add LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY to your .env file."
    )
    langfuse_handler = None
else:
    try:
        langfuse_handler = CallbackHandler()
        logger.info(f"Langfuse initialized. Sending traces to: {langfuse_host}")
    except Exception as e:
        logger.error(f"Failed to initialize Langfuse: {e}")
        langfuse_handler = None

def get_langfuse_config() -> dict:
    """Return LangChain config dict with Langfuse callbacks for tracing."""
    if langfuse_handler:
        return {"callbacks": [langfuse_handler]}
    return {}

def flush_langfuse_traces():
    """Force flush all pending traces to Langfuse. Call this before the app exits or on errors."""
    if langfuse_handler:
        try:
            langfuse_handler.flush()
            logger.info("Langfuse traces flushed")
        except Exception as e:
            logger.warning(f"Failed to flush Langfuse traces: {e}")
