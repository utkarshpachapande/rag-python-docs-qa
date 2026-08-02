import os
from langchain_chroma import Chroma
from src.embeddings import get_embedding_model

CHROMA_PATH = "vectorstore/chroma_db"

def build_vector_db(chunks):
    """Builds and saves the Chroma Vector Database."""
    print("Building Vector DB... (Generating Embeddings)")
    embeddings = get_embedding_model()
    
    # Create persistent Chroma DB
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    return vector_db

def load_vector_db():
    """Loads the existing Vector DB from disk."""
    if not os.path.exists(CHROMA_PATH):
        return None
    embeddings = get_embedding_model()
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)