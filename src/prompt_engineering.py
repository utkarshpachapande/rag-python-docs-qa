from langchain_core.prompts import ChatPromptTemplate

def get_rag_prompt():
    """Structured prompt to force grounded answers."""
    template = """
    You are an expert AI/ML Python coding assistant.
    Use the following pieces of retrieved context to answer the user's question.
    If you don't know the answer based on the context, say "I cannot answer this based on the provided documentation." DO NOT hallucinate.
    
    Context:
    {context}
    
    Chat History:
    {history}
    
    Question: {question}
    
    Provide a clear, code-focused answer:
    """
    return ChatPromptTemplate.from_template(template)