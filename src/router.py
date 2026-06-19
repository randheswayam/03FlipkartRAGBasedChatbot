from semantic_router import Route
from semantic_router.routers import SemanticRouter
from semantic_router.encoders.local import LocalEncoder
from semantic_router.index.local import LocalIndex

# 1. Define Routes
faq_route = Route(
    name="faq",
    utterances=[
        "what is the return policy of the products?",
        "Do I get discount with the HDFC credit card?",
        "How can I track my order?",
        "What payment methods are accepted?",
        "How long does it take to process a refund?",
        "Are there any ongoing sales or promotions?",
        "Can I cancel or modify my order after placing it?",
        "Do you offer international shipping?",
        "What should I do if I receive a damaged product?",
        "How do I use a promo code during checkout?",
        "returns refunds refund cancellations status order track",
        "can i cancel this product",
        "how to get a refund",
        "who do i contact for help",
        "escalate to human customer service support"
    ]
)

product_route = Route(
    name="product_inquiry",
    utterances=[
        "show me smartphones under 20,000",
        "which laptop has 16 GB RAM",
        "what is the rating of product X?",
        "find running shoes for women",
        "price of campus shoes",
        "discount on aadi mesh shoes",
        "show me shoes under 1000",
        "list some brands",
        "what categories of products do you have?",
        "do you have nike shoes in stock?",
        "check if you have shoes from puma",
        "search product database price rating brand stock category url",
        "show me the cheapest shoes",
        "highly rated sneakers",
        "shoes with discount"
    ]
)

routes = [faq_route, product_route]

# 2. Initialize Local Encoder using Sentence Transformers
encoder = LocalEncoder(name="sentence-transformers/all-MiniLM-L6-v2")

# 3. Initialize LocalIndex with auto_sync="local"
index = LocalIndex(auto_sync="local")

# 4. Create SemanticRouter
router = SemanticRouter(
    encoder=encoder,
    routes=routes,
    index=index,
    auto_sync="local"
)

def classify_route(query_text: str) -> str:
    """Classifies user query into 'faq', 'product_inquiry', or 'fallback'."""
    try:
        # Get decision from semantic-router
        res = router(query_text)
        
        # Check score threshold to determine confidence
        # A route matches if it is returned and its score is >= 0.25 (typical embedding similarity)
        if res and res.name and res.similarity_score and res.similarity_score >= 0.25:
            return res.name
            
        return "fallback"
    except Exception as e:
        # Fallback in case of errors
        print(f"Error during route classification: {e}")
        return "fallback"
