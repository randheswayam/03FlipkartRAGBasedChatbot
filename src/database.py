import sqlite3
import re
from src.config import DB_PATH

def validate_sql(sql_query: str) -> bool:
    """Validates that the SQL query is a read-only SELECT statement and contains no harmful operations."""
    cleaned = sql_query.strip().lower()
    
    # Must start with SELECT (ignoring leading whitespace and comments)
    # Match optional SQL comments at the beginning e.g., /* comment */
    cleaned_no_comments = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL).strip()
    cleaned_no_comments = re.sub(r'--.*$', '', cleaned_no_comments, flags=re.MULTILINE).strip()
    
    if not cleaned_no_comments.startswith("select"):
        return False
        
    # Block list of writing commands
    block_words = ["insert", "update", "delete", "drop", "alter", "replace", "create", "truncate", "grant", "revoke"]
    for word in block_words:
        # Use word boundaries to check for actual command keywords
        pattern = r'\b' + re.escape(word) + r'\b'
        if re.search(pattern, cleaned):
            return False
            
    return True

def query_products(sql_query: str):
    """Executes a read-only SELECT query against the products SQLite database.
    
    Returns a tuple: (list of dicts containing rows, error_message_string)
    """
    if not validate_sql(sql_query):
        return None, "Invalid or unauthorized SQL query. Only SELECT statements are permitted."
        
    try:
        # Connect in read-only mode using uri=True
        # D:\08-06-2026\03FlipkartRAGBasedChatbot\db.sqlite
        # We need absolute path for file URI
        import os
        abs_db_path = os.path.abspath(DB_PATH).replace('\\', '/')
        conn = sqlite3.connect(f"file:{abs_db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        
        cursor = conn.cursor()
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        
        # Convert to list of dicts
        results = [dict(row) for row in rows]
        
        cursor.close()
        conn.close()
        
        return results, None
    except Exception as e:
        return None, str(e)
