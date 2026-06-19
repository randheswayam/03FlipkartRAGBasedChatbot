import chromadb
from chromadb.utils import embedding_functions
from src.config import CHROMA_PATH

class FAQRetriever:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.collection = self.chroma_client.get_collection(name="faqs", embedding_function=self.emb_fn)
        
    def query_faqs(self, query_text: str, n_results: int = 3):
        """Queries ChromaDB FAQ collection for the top N matches.
        
        Returns:
            list of dicts containing 'question', 'answer', 'category', and 'distance'.
        """
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            
            # Format outputs
            formatted = []
            if results and 'documents' in results and results['documents']:
                docs = results['documents'][0]
                metas = results['metadatas'][0]
                distances = results['distances'][0] if 'distances' in results and results['distances'] else [0.0]*len(docs)
                ids = results['ids'][0]
                
                for idx in range(len(docs)):
                    formatted.append({
                        'id': ids[idx],
                        'question': docs[idx],
                        'answer': metas[idx].get('answer', ''),
                        'category': metas[idx].get('category', 'general'),
                        'distance': distances[idx]
                    })
            return formatted, None
        except Exception as e:
            return [], str(e)
