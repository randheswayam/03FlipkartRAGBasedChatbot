import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import sqlite3
import hashlib
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from pandasql import sqldf

# Import config
from src.config import DB_PATH, CHROMA_PATH, FAQ_CSV_PATH, PRODUCT_CSV_PATH

def generate_mock_stock(url):
    """Generates a deterministic mock stock level between 5 and 50 based on URL hash."""
    if not url or pd.isna(url):
        return 10
    hash_val = int(hashlib.md5(url.encode('utf-8')).hexdigest(), 16)
    return 5 + (hash_val % 46)

def ingest_faqs():
    print("--- Ingesting FAQ Data into ChromaDB ---")
    if not os.path.exists(FAQ_CSV_PATH):
        raise FileNotFoundError(f"FAQ CSV file not found at {FAQ_CSV_PATH}")
    
    # Load FAQ data
    df_faq = pd.read_csv(FAQ_CSV_PATH)
    print(f"Loaded FAQ shape: {df_faq.shape}")
    
    # Clean FAQ data
    df_faq.dropna(subset=['question', 'answer'], inplace=True)
    df_faq['question'] = df_faq['question'].str.strip()
    df_faq['answer'] = df_faq['answer'].str.strip()
    df_faq.drop_duplicates(subset=['question'], inplace=True)
    print(f"Cleaned FAQ shape: {df_faq.shape}")
    
    # Initialize ChromaDB client and embedding function
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    # Get or create collection
    # Use overwrite pattern: delete collection if exists to refresh, or get it
    try:
        chroma_client.delete_collection(name="faqs")
        print("Deleted existing FAQ collection to re-ingest.")
    except Exception:
        pass
        
    collection = chroma_client.create_collection(name="faqs", embedding_function=emb_fn)
    
    # Prepare documents, metadatas, ids
    ids = [str(i) for i in range(len(df_faq))]
    documents = df_faq['question'].tolist()
    # Store answer and category (if exists) in metadata
    metadatas = []
    for _, row in df_faq.iterrows():
        meta = {'answer': row['answer']}
        if 'category' in row and not pd.isna(row['category']):
            meta['category'] = str(row['category'])
        else:
            meta['category'] = 'general'
        metadatas.append(meta)
        
    # Ingest into ChromaDB
    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"Successfully ingested {len(documents)} FAQs into ChromaDB collection.")

def ingest_products():
    print("\n--- Ingesting Product Data into SQLite ---")
    if not os.path.exists(PRODUCT_CSV_PATH):
        raise FileNotFoundError(f"Product CSV file not found at {PRODUCT_CSV_PATH}")
        
    # Load primary Product data
    df_products = pd.read_csv(PRODUCT_CSV_PATH)
    print(f"Loaded primary products shape: {df_products.shape}")
    
    # Check for additional scraped product data in web-scrapping folder
    scraped_csv = "web-scrapping/flipkart_product_data.csv"
    if os.path.exists(scraped_csv):
        try:
            df_scraped = pd.read_csv(scraped_csv)
            print(f"Loaded scraped products shape: {df_scraped.shape}")
            df_products = pd.concat([df_products, df_scraped], ignore_index=True)
            print(f"Combined products shape: {df_products.shape}")
        except Exception as e:
            print(f"Warning: Could not load scraped products from {scraped_csv}: {e}")
    
    # Clean and transform
    df_products.dropna(subset=['title', 'price'], inplace=True)
    
    # Map fields to match requirement: name, brand, category, price, rating, stock, and URL
    # Raw columns: product_link,title,brand,price,discount,avg_rating,total_ratings
    df_products['url'] = df_products['product_link'].str.strip()
    df_products['product_name'] = df_products['title'].str.strip()
    df_products['brand'] = df_products['brand'].fillna('Unknown').str.strip()
    df_products['category'] = 'Sports Shoes' # All products scraped are sports shoes for women
    
    # Handle numeric columns
    df_products['price'] = pd.to_numeric(df_products['price'], errors='coerce').fillna(0).astype(int)
    df_products['discount'] = pd.to_numeric(df_products['discount'], errors='coerce').fillna(0.0)
    df_products['rating'] = pd.to_numeric(df_products['avg_rating'], errors='coerce').fillna(0.0)
    df_products['total_ratings'] = pd.to_numeric(df_products['total_ratings'], errors='coerce').fillna(0).astype(int)
    
    # Generate mock stock
    df_products['stock'] = df_products['url'].apply(generate_mock_stock)
    
    # Clean duplicates
    df_products.drop_duplicates(subset=['product_name', 'price'], inplace=True)
    
    # Keep only necessary fields
    clean_cols = ['url', 'product_name', 'brand', 'category', 'price', 'discount', 'rating', 'total_ratings', 'stock']
    df_clean = df_products[clean_cols].copy()
    print(f"Cleaned products shape: {df_clean.shape}")
    
    # Exploratory validation using pandasql
    print("Running data validation using pandasql...")
    
    # Validation query: Average price and count by brand
    validation_q = """
        SELECT brand, COUNT(*) as product_count, ROUND(AVG(price), 2) as avg_price, MAX(rating) as max_rating
        FROM df_clean
        GROUP BY brand
        ORDER BY product_count DESC
        LIMIT 5
    """
    validation_res = sqldf(validation_q, env=locals())
    print("Top Brands validation statistics:")
    print(validation_res)
    
    # Load to SQLite
    os.makedirs(os.path.dirname(DB_PATH) if os.path.dirname(DB_PATH) else '.', exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Drop table if exists to refresh
    cursor.execute("DROP TABLE IF EXISTS products;")
    cursor.execute("DROP TABLE IF EXISTS product;") # also drop old table name if existing
    
    # Create product tables
    cursor.execute('''
    CREATE TABLE products (
        url TEXT,
        product_name TEXT,
        brand TEXT,
        category TEXT,
        price INTEGER,
        discount REAL,
        rating REAL,
        total_ratings INTEGER,
        stock INTEGER
    );
    ''')
    conn.commit()
    
    # Write dataframe to SQL
    df_clean.to_sql('products', conn, if_exists='append', index=False)
    # Also write to 'product' table name for backwards compatibility
    df_clean.to_sql('product', conn, if_exists='replace', index=False)
    
    conn.close()
    print(f"Successfully ingested {len(df_clean)} products into SQLite database '{DB_PATH}'.")

if __name__ == "__main__":
    ingest_faqs()
    ingest_products()
    print("\n--- Ingestion Pipeline Completed Successfully! ---")
