import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
DB_PATH = os.getenv("DB_PATH", "db.sqlite")
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db_store")
FAQ_CSV_PATH = os.getenv("FAQ_CSV_PATH", "app/resources/faq_data.csv")
PRODUCT_CSV_PATH = os.getenv("PRODUCT_CSV_PATH", "app/resources/ecommerce_data_final.csv")
