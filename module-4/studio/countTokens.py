import tiktoken

def count_tokens(text, model="gpt-3.5-turbo"):
    """Count the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)

def truncate_to_token_limit(text, max_tokens=1000, model="gpt-3.5-turbo"):
    """Truncate text to stay within token limit."""
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    
    if len(tokens) <= max_tokens:
        return text
    
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)

def estimate_completion_tokens(prompt_tokens, max_output_tokens=500):
    """Estimate total tokens (prompt + completion) for budgeting."""
    return prompt_tokens + max_output_tokens

def track_token_usage(prompt, response, model="gpt-3.5-turbo"):
    """Track token usage for a complete interaction."""
    prompt_tokens = count_tokens(prompt, model)
    response_tokens = count_tokens(response, model)
    total_tokens = prompt_tokens + response_tokens
    
    costs = {
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "gpt-4": {"input": 0.03, "output": 0.06}
    }
    
    # Calculate cost in USD per 1K tokens
    input_cost = (prompt_tokens / 1000) * costs[model]["input"]
    output_cost = (response_tokens / 1000) * costs[model]["output"]
    total_cost = input_cost + output_cost
    
    return {
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "total_tokens": total_tokens,
        "cost_usd": total_cost
    }

# Example usage
long_text = """This is a very long document that might exceed our token limits.
               It contains many paragraphs of text that would need to be processed
               by the LLM, but we want to ensure we don't go over our budget...""" * 50

# Count tokens
token_count = count_tokens(long_text)
print(f"Original text has {token_count} tokens")

# Truncate to limit
truncated_text = truncate_to_token_limit(long_text, max_tokens=1000)
truncated_token_count = count_tokens(truncated_text)
print(f"Truncated text has {truncated_token_count} tokens")

# Budget estimation for a conversation
user_message = "Can you explain quantum computing to me?"
prompt_tokens = count_tokens(user_message)
total_expected = estimate_completion_tokens(prompt_tokens)
print(f"Expected total tokens for this interaction: {total_expected}")

# Track actual usage after getting a response
mock_response = "Quantum computing is a type of computing that uses quantum mechanics..."
usage_stats = track_token_usage(user_message, mock_response)
print(f"Actual usage: {usage_stats}")