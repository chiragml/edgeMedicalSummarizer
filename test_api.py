"""
Start the medical AI inference API
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import and run
from ml_scripts.inference import app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)