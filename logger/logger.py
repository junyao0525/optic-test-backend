import logging
import sys

# Configure once
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Export logger
logger = logging.getLogger("fastapi_app")  # consistent app-wide logger name
