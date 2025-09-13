import anthropic

def get_llm_client():
    """
    Returns a client for the LLM. Use the mock client for testing.
    For production, replace `MockClaudeClient()` with `Anthropic()`.
    """
    # Replace with `anthropic.Anthropic()` and your API key in a real application.
    client = anthropic.Anthropic(api_key="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")
    return client
