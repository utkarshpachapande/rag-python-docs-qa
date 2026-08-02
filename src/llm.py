from langchain_openai import ChatOpenAI

def get_llm(model_type="Groq (Llama 3)", api_key=None):
    """Returns the selected LLM based on user constraints."""
    if model_type == "Groq (Llama 3)" and api_key:
        # Free, fast cloud inference - works anywhere, no local install needed.
        # OpenAI-compatible endpoint, so we reuse ChatOpenAI with Groq's base_url.
        return ChatOpenAI(
            model="llama-3.1-8b-instant",
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=0.2,
        )
    elif model_type == "OpenAI API" and api_key:
        return ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key, temperature=0.2)
    elif model_type == "Ollama (Local)":
        # Only works when Ollama is installed and running locally.
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(model="llama3:latest", temperature=0.2)
    else:
        raise ValueError("Invalid LLM configuration. Check your API key or model selection.")