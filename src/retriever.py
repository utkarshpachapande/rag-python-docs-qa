def get_retriever(vector_db, k=3):
    """Creates a retriever interface with configurable top-k."""
    return vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )