"""
Start the medical AI inference API
"""
import os
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create logs directory if it doesn't exist
logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Create log filename with timestamp
log_filename = f"api_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_filepath = os.path.join(logs_dir, log_filename)

# Create empty log file
with open(log_filepath, 'w') as f:
    f.write(f"API Server Log - Started at {datetime.now().isoformat()}\n")
    f.write("=" * 50 + "\n")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler()  # Also log to console
    ]
)

logger = logging.getLogger(__name__)
logger.info(f"Log file created at: {log_filepath}")

# Import and run
from ml_scripts.inference import app

if __name__ == "__main__":
    logger.info("Starting Flask server on port 5001")
    app.run(host="0.0.0.0", port=5001, debug=True)