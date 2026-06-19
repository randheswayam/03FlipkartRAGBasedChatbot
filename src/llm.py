import re
from groq import Groq
from src.config import GROQ_API_KEY, GROQ_MODEL

# Initialize Groq client
# If GROQ_API_KEY is not set (e.g., during initialization/template step), we can handle it gracefully.
client = None
if GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here":
    client = Groq(api_key=GROQ_API_KEY)

def get_client():
    global client
    if client is None:
        # Re-try loading in case it was updated in the environment
        import os
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if api_key and api_key != "your_groq_api_key_here":
            client = Groq(api_key=api_key)
        else:
            raise ValueError("Groq API Key is not set or is still the default placeholder in .env. Please set GROQ_API_KEY in your .env file.")
    return client

def generate_sql_query(user_query: str) -> str:
    """Uses Groq Llama 3.3 to convert a natural language product search into an SQLite query."""
    groq_client = get_client()
    
    system_prompt = """You are a Text-to-SQL translation assistant. Your goal is to translate a user's natural language query into a valid, read-only SQLite SELECT statement for a table named 'products'.

Table Schema:
- url: TEXT (Link to the product page on Flipkart)
- product_name: TEXT (Full title of the product)
- brand: TEXT (Brand of the product, e.g., 'aadi', 'CAMPUS', 'Sparx', 'asian', 'PUMA', 'ADIDAS', 'Skechers')
- category: TEXT (Product category, e.g., 'Sports Shoes')
- price: INTEGER (Price in Indian Rupees)
- discount: REAL (Discount percentage as a decimal fraction, e.g. 0.73 means 73% off, 0.0 means no discount)
- rating: REAL (Average customer rating out of 5.0, e.g., 4.2)
- total_ratings: INTEGER (Total number of customer ratings/reviews, e.g., 24914)
- stock: INTEGER (Number of units available in stock, e.g., 12)

SQL Rules:
1. ONLY return a valid SQLite SELECT query.
2. Do NOT output any markdown tags (like ```sql), explanation, or other text. Return ONLY the raw SQL string.
3. For text search on product name, use: LIKE '%term%'
4. For brand matching, do a case-insensitive check (e.g., LOWER(brand) = 'adidas' or brand LIKE '%adidas%').
5. Price filters: e.g., 'under 1000' -> price < 1000; 'between 500 and 1000' -> price BETWEEN 500 AND 1000.
6. Rating filters: e.g., 'rating above 4' -> rating >= 4.0.
7. Availability/stock filters: e.g., 'in stock' -> stock > 0.
8. Sort order: If the user asks for 'best', 'highest rated', or 'top', order by rating DESC and total_ratings DESC. If they ask for 'cheapest' or 'lowest price', order by price ASC.
9. ALWAYS include a 'LIMIT 5' clause to keep the result set concise.
10. Ensure the query returns all relevant columns: url, product_name, brand, price, discount, rating, total_ratings, stock.

Examples:
User: "show me campus shoes under 1000"
SQL: SELECT url, product_name, brand, price, discount, rating, total_ratings, stock FROM products WHERE brand LIKE '%campus%' AND price < 1000 LIMIT 5

User: "which sneakers are highly rated"
SQL: SELECT url, product_name, brand, price, discount, rating, total_ratings, stock FROM products WHERE rating >= 4.2 ORDER BY rating DESC, total_ratings DESC LIMIT 5
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Translate this query to SQL: {user_query}"}
    ]
    
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=200
    )
    
    sql = response.choices[0].message.content.strip()
    
    # Strip markdown block formatting if the model still generated it
    sql = re.sub(r'^```sql\s*', '', sql, flags=re.IGNORECASE)
    sql = re.sub(r'^```\s*', '', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\s*```$', '', sql, flags=re.IGNORECASE)
    
    return sql

def generate_response(user_query: str, route: str, context: str) -> str:
    """Generates a final natural-language response based on user query, route, and retrieved context."""
    groq_client = get_client()
    
    if route == "faq":
        system_prompt = """You are a helpful Flipkart Customer Support assistant.
Your goal is to answer the user's questions using ONLY the provided FAQ context.
Ground your response strictly in the facts provided in the FAQ context.
Keep your response concise, professional, and friendly (maximum 2-3 sentences).
If the context does not contain the answer, politely tell the user you don't know the answer and guide them to official support channels.
"""
        user_prompt = f"FAQ Context:\n{context}\n\nUser Question: {user_query}\nAnswer:"
        
    elif route == "product_inquiry":
        system_prompt = """You are a helpful Flipkart Sales assistant.
Your goal is to answer product questions based on the retrieved product database records.
Format the products cleanly using bullet points, and include the price, rating, and brand.
Provide clickable markdown links for the products.
The link markdown should be in the format: [Product Name](url)
Include a brief, friendly summary. If no products are found in the records, suggest that they search for other keywords or brand names.
Keep your answer clear, helpful, and concise.
"""
        user_prompt = f"Product Database Records:\n{context}\n\nUser Question: {user_query}\nResponse:"
        
    else:  # fallback
        system_prompt = """You are a polite Flipkart Customer Support assistant.
The user is asking a question that is out-of-scope or unsupported by our automated catalog and standard FAQ search.
Provide a courteous fallback response.
Explain clearly that you cannot answer this specific question, and guide them to the official support options:
1. Go to the 'Account' tab and select 'Help Centre' in the Flipkart App or Website.
2. Use the 'Need Help' chat option within their order page for order-specific assistance.
3. Call the Flipkart Customer Support 24x7 helpline.
Keep the response professional, friendly, and structured.
"""
        user_prompt = f"User Question: {user_query}\nResponse:"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=600
    )
    
    return response.choices[0].message.content.strip()
