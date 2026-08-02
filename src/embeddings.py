from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embedding_model():
    """Loads a fast, low-memory embedding model perfect for 8GB RAM."""
    # all-MiniLM-L6-v2 is highly efficient and runs well on Intel i5 CPUs
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'} 
    )