import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Template: consider using secret manager in real projects.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

def set_environment():
    for k, v in globals().items():
        if k.endswith("_API_KEY") and v:
            os.environ[k] = v
