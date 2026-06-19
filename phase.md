Phase 1: Chatbot Development in Streamlit will cover building a Streamlit-based chatbot for Flipkart user support, with routing between FAQ retrieval from ChromaDB and product/pricing lookup from SQLite using a semantic router. The stack is consistent with common implementations that combine Streamlit for UI, ChromaDB for vector retrieval, SQLite for structured product data, Sentence Transformers for embeddings, Groq for LLM responses, and Semantic Router for intent classification.

Objective
Build a chatbot in Streamlit with core support for:

FAQ handling through vector search in ChromaDB.

Product inquiries through SQLite-based structured retrieval.

Intent routing using semantic-router[local].

Response generation using the Groq-hosted Llama 3.3 model.

Secure configuration management through .env variables.

Updated deliverables
1. Environment and dependency setup
Set up the Python environment and install all required packages for UI, retrieval, routing, embeddings, structured querying, and secrets management. The semantic-router project specifically documents semantic-router[local] for fully local routing components, and .env-based API key handling is a common setup pattern for Streamlit chatbot apps.

Required libraries

streamlit – chatbot frontend and interaction layer.

pandas – loading, cleaning, and transforming FAQ and product datasets.

pandasql – SQL-style querying over pandas DataFrames during preprocessing, validation, or lightweight analysis before SQLite loading.

python-dotenv – loading Groq API keys and configuration variables from .env.

groq – LLM access for response generation using Llama 3.3.

semantic-router[local] – local semantic intent routing for FAQ vs product query classification.

chromadb – vector database for FAQ storage and retrieval.

sentence_transformers – embedding generation for FAQ vectorization and semantic matching.

2. FAQ data ingestion into ChromaDB
Prepare the FAQ dataset using pandas, clean and structure question-answer pairs, generate embeddings with sentence_transformers, and store them in chromadb for semantic retrieval. This matches the typical RAG pattern described in Streamlit and ChromaDB chatbot/search implementations.

Sub-steps to add

Load FAQ CSV/JSON data with pandas.

Clean duplicates, null values, and inconsistent formatting before indexing.

Convert FAQ questions or chunks into embeddings using sentence_transformers.

Insert vectors, metadata, and answers into a ChromaDB collection for retrieval.

Validate retrieval quality with sample user questions in the Streamlit app.

3. Product and pricing pipeline with SQLite
Store product catalog and pricing information in SQLite and use pandas for ingestion and validation before database insertion. Existing examples of NL-to-SQL chatbot patterns also use SQLite as the structured source for product or business data.

Sub-steps to add

Load product data using pandas.

Use pandasql for exploratory SQL-style validation on DataFrames before pushing clean records into SQLite.

Create SQLite tables for product name, brand, category, price, rating, stock, and URL fields.

Support filtered product lookup and pricing queries from the chatbot.

Keep database access read-only from the chatbot flow for safety.

Route support
FAQs route
The FAQs route will classify support-style questions and retrieve the most relevant answers from ChromaDB using semantic similarity. This architecture is directly aligned with semantic-router-based query classification plus ChromaDB-backed FAQ retrieval.

Product inquiries route
The product inquiries route will classify catalog or price-related questions and fetch structured results from SQLite, optionally using the LLM to format the final answer in natural language. Examples in the sources describe this exact split between vector retrieval for FAQs and SQL/SQLite for structured data.

Streamlit frontend
Build a Streamlit interface with chat history, user input box, route-aware response rendering, and sections for retrieved FAQ context or product results. Streamlit is widely used in the referenced chatbot and search implementations as the frontend layer for interactive LLM applications.

Frontend steps to add

Chat input/output window in Streamlit.

Session-based chat history handling.

Route label or debug mode to show whether the query hit FAQ or product search during testing.

Display of retrieved FAQ source text or product records before final LLM answer generation.