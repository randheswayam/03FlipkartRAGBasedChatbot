import streamlit as st
import pandas as pd
import json
import os

# Import modules from src
from src.config import GROQ_API_KEY
from src.router import classify_route
from src.vector_store import FAQRetriever
from src.database import query_products
from src.llm import generate_sql_query, generate_response

# Page Configuration
st.set_page_config(
    page_title="Flipkart AI Assistant",
    page_icon="🛍️",
    layout="wide"
)

# Custom Flipkart Styling (Aesthetics)
st.markdown("""
<style>
    /* Main Background and Fonts */
    .stApp {
        background-color: #f1f3f6;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    /* Header Customization */
    .header-container {
        background: linear-gradient(135deg, #2874f0 0%, #1e56b3 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .header-subtitle {
        font-size: 1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    .brand-yellow {
        color: #ffe500;
        font-weight: 900;
    }
    
    /* Chat Bubble Styling */
    .chat-bubble {
        padding: 1rem 1.25rem;
        border-radius: 16px;
        margin-bottom: 1rem;
        max-width: 75%;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        line-height: 1.5;
        position: relative;
    }
    .user-bubble {
        background-color: #2874f0;
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }
    .assistant-bubble {
        background-color: white;
        color: #212121;
        margin-right: auto;
        border-bottom-left-radius: 4px;
        border: 1px solid #e0e0e0;
    }
    .route-badge {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        position: absolute;
        top: -10px;
        left: 15px;
        color: white;
    }
    .route-faq {
        background-color: #26a541; /* Green */
    }
    .route-product {
        background-color: #ff9f00; /* Yellow-Orange */
    }
    .route-fallback {
        background-color: #d32f2f; /* Red */
    }
    
    /* Custom input box border and button colors */
    div.stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
    }
    div.stTextInput > div > div > input:focus {
        border-color: #2874f0;
        box-shadow: 0 0 0 2px rgba(40,116,240,0.2);
    }
    
    /* Sidebar Card Layout */
    .sidebar-card {
        background-color: white;
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Try initializing retriever
@st.cache_resource
def get_faq_retriever():
    try:
        return FAQRetriever()
    except Exception as e:
        st.error(f"Error initializing FAQ database: {e}. Ensure you have run 'python ingest.py' first.")
        return None

retriever = get_faq_retriever()

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar configuration
with st.sidebar:
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.image("https://img1a.flixcart.com/www/linchpin/fk-cp-zion/img/flipkart-plus_8d85f4.png", width=120)
    st.markdown("### Bot Configuration")
    
    # Check Groq Key Status
    is_key_valid = GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here"
    if is_key_valid:
        st.success("🟢 Groq API Connected")
    else:
        st.error("🔴 Groq API Key Missing")
        st.info("Please set your `GROQ_API_KEY` in the `.env` file in the root directory.")
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Settings Card
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown("### Settings")
    debug_mode = st.checkbox("🔍 Developer / Debug Mode", value=True, help="Shows classification results, generated SQL, and vector matches.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sample Questions
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown("### Sample FAQ Queries")
    st.caption("• What is the return policy?")
    st.caption("• Can I cancel my order?")
    st.caption("• Do you accept UPI?")
    st.caption("• How can I track my order?")
    
    st.markdown("### Sample Catalog Queries")
    st.caption("• Show me shoes under 1000")
    st.caption("• Which brand has rating above 4.5?")
    st.caption("• Puma running shoes")
    st.caption("• Do you have Campus shoes in stock?")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Action button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main Header Design
st.markdown("""
<div class="header-container">
    <div>
        <h1 class="header-title">Flipkart <span class="brand-yellow">SmartSupport</span> AI</h1>
        <p class="header-subtitle">Phase 1 Demo: Intelligent FAQ Retrieval & Structured Catalog Inquiry</p>
    </div>
    <div style="text-align: right;">
        <span style="font-size: 0.85rem; background: rgba(255,255,255,0.2); padding: 0.4rem 0.8rem; border-radius: 20px;">
            Powered by Llama 3.3 & ChromaDB
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# Display Chat History
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-bubble user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        # Show route badge if debug is enabled
        badge_html = ""
        if debug_mode and "debug_info" in msg:
            route = msg["debug_info"].get("route", "unknown")
            if route == "faq":
                badge_html = '<div class="route-badge route-faq">🏷️ FAQ Match</div>'
            elif route == "product_inquiry":
                badge_html = '<div class="route-badge route-product">🛍️ Catalog Search</div>'
            else:
                badge_html = '<div class="route-badge route-fallback">⚠️ Fallback</div>'
                
        st.markdown(f'<div class="chat-bubble assistant-bubble">{badge_html}{msg["content"]}</div>', unsafe_allow_html=True)
        
        # Show debug expander if checked and debug_info is available
        if debug_mode and "debug_info" in msg:
            debug_data = msg["debug_info"]
            with st.expander("🛠️ Developer Details", expanded=False):
                st.write(f"**Detected Intent Route:** `{debug_data.get('route')}`")
                if debug_data.get("route") == "faq":
                    st.write("**Top ChromaDB Matches:**")
                    st.json(debug_data.get("matches", []))
                elif debug_data.get("route") == "product_inquiry":
                    st.code(f"-- Generated SQL Query --\n{debug_data.get('generated_sql', '')}", language="sql")
                    if debug_data.get("sql_error"):
                        st.error(f"SQL Error: {debug_data.get('sql_error')}")
                    else:
                        st.write(f"**Retrieved Records ({len(debug_data.get('records', []))} items):**")
                        st.json(debug_data.get("records", []))

# User input text box
user_input = st.chat_input("Ask about order policies, payment methods, or search our product catalog...")

if user_input:
    # 1. Display User Message
    st.markdown(f'<div class="chat-bubble user-bubble">{user_input}</div>', unsafe_allow_html=True)
    
    # Check if API is connected
    if not is_key_valid:
        st.markdown("""
        <div class="chat-bubble assistant-bubble" style="color: #d32f2f; border-color: #d32f2f;">
            ❌ <strong>Error:</strong> Groq API key is not configured. 
            Please configure your <code>GROQ_API_KEY</code> in the <code>.env</code> file in your workspace to enable response generation.
        </div>
        """, unsafe_allow_html=True)
        st.stop()
        
    with st.spinner("Processing inquiry..."):
        # 2. Classify Route
        route = classify_route(user_input)
        
        debug_info = {"route": route}
        final_response = ""
        
        # 3. Handle Routes
        if route == "faq":
            if retriever:
                matches, error = retriever.query_faqs(user_input, n_results=3)
                debug_info["matches"] = matches
                
                if error:
                    st.error(f"ChromaDB Query Error: {error}")
                    context = ""
                else:
                    # Format matches for LLM prompt
                    context_parts = []
                    for match in matches:
                        context_parts.append(f"Q: {match['question']}\nA: {match['answer']}")
                    context = "\n\n".join(context_parts)
            else:
                context = ""
                
            final_response = generate_response(user_input, "faq", context)
            
        elif route == "product_inquiry":
            # Generate SQL Query via LLM
            sql_query = generate_sql_query(user_input)
            debug_info["generated_sql"] = sql_query
            
            # Execute SQL Query
            records, sql_error = query_products(sql_query)
            debug_info["records"] = records
            debug_info["sql_error"] = sql_error
            
            # Generate final response based on database output
            if sql_error:
                context = f"Error querying database: {sql_error}"
            elif not records or len(records) == 0:
                context = "No products found matching the criteria."
            else:
                context = json.dumps(records, indent=2)
                
            final_response = generate_response(user_input, "product_inquiry", context)
            
        else: # Fallback / Out-of-scope
            final_response = generate_response(user_input, "fallback", "")
            
        # Add assistant response to history
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        st.session_state.messages.append({
            "role": "assistant",
            "content": final_response,
            "debug_info": debug_info
        })
        
        st.rerun()
