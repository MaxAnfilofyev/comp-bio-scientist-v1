import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Default database URL
DB_NAME = "ai_scientist.sqlite"
DB_PATH = os.path.join(BASE_DIR, DB_NAME)
DATABASE_URL = f"sqlite:///{DB_PATH}"
