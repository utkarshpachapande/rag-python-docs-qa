from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(documents):
    """Splits text into manageable chunks for the Vector DB."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150, # Crucial for maintaining context between chunks
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} documents.")
    return chunks