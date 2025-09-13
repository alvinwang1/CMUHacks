import anthropic

def get_llm_client():
    """
    Returns a client for the LLM. Use the mock client for testing.
    For production, replace `MockClaudeClient()` with `Anthropic()`.
    """
    # Replace with `anthropic.Anthropic()` and your API key in a real application.
    client = anthropic.Anthropic(api_key="")
    return client
