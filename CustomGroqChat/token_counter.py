"""
Token Counter for Groq Cloud API client.

Provides accurate token counting for requests using tiktoken.
"""
import json
import tiktoken
from typing import Dict, Any, List, Union, Optional

# Default encoding to use if model is not found
DEFAULT_ENCODING = "cl100k_base"


def count_tokens_in_message(message: Dict[str, str], encoding: tiktoken.Encoding) -> int:
    """
    Count tokens in a single message.

    Args:
        message: A message dictionary with 'role' and 'content' keys
        encoding: The tokenizer encoding to use

    Returns:
        Number of tokens in the message
    """
    content = message.get("content", "")                                                                                # Find content in message

    # Count tokens in the content
    content_tokens = len(encoding.encode(content))                                                                      # Count tokens in content

    # Each message follows the format: <role>content, with role being user, assistant, or system
    # We add a small constant for the message format (based on approximation)
    format_tokens = 4  # Approximation for message formatting                                                           # Format tokens (e.g., <role>)

    # Function calls or additional fields would add more tokens
    if "function_call" in message:                                                                                      # Check for function call
        function_call = message["function_call"]                                                                        # Get function call
        if isinstance(function_call, dict):                                                                             # Check if function call is a dictionary
            # Count tokens in the function call
            function_call_str = json.dumps(function_call)                                                               # Convert function call to string
            function_tokens = len(encoding.encode(function_call_str))
            content_tokens += function_tokens

    return content_tokens + format_tokens


def count_tokens_in_messages(messages: List[Dict[str, str]], model_name: str) -> int:
    """
    Count tokens in a list of messages.

    Args:
        messages: List of message dictionaries
        model_name: Name of the model to use for token counting

    Returns:
        Total number of tokens in the messages
    """
    if not messages:
        return 0

    encoding = DEFAULT_ENCODING
    token_count = 0

    # Add tokens for each message
    for message in messages:
        token_count += count_tokens_in_message(message, encoding)

    # Add tokens for the overall formatting
    # Most chat models add a few tokens for formatting
    token_count += 3  # Approximation for overall formatting

    return token_count


def count_tokens_in_prompt(prompt: str, model_name: str) -> int:
    """
    Count tokens in a text prompt.

    Args:
        prompt: Text prompt
        model_name: Name of the model to use for token counting

    Returns:
        Number of tokens in the prompt
    """
    encoding = DEFAULT_ENCODING
    return len(encoding.encode(prompt))


def count_tokens_in_request(request_data: Dict[str, Any], model_name: str) -> int:
    """
    Count tokens in a complete request.

    Args:
        request_data: Request data dictionary
        model_name: Name of the model

    Returns:
        Total number of tokens in the request
    """
    # Check for chat messages
    if "messages" in request_data:
        return count_tokens_in_messages(request_data["messages"], model_name)

    # Check for text prompt
    if "prompt" in request_data:
        return count_tokens_in_prompt(request_data["prompt"], model_name)

    # Unknown request format
    # Estimate a default value
    return 10


def estimate_completion_tokens(request_data: Dict[str, Any], default_tokens: int = 100) -> int:
    """
    Estimate the number of tokens in the completion based on request parameters.

    Args:
        request_data: Request data dictionary
        default_tokens: Default value if no max_tokens is specified

    Returns:
        Estimated number of completion tokens
    """
    # Get max_tokens from request or default value
    max_tokens = request_data.get("max_tokens", default_tokens)

    # If max_tokens is 0 or negative, use default
    if max_tokens <= 0:
        max_tokens = default_tokens

    return max_tokens


def count_request_and_completion_tokens(request_data: Dict[str, Any], model_name: str) -> Dict[str, int]:
    """
    Count tokens for both the request and the estimated completion.

    Args:
        request_data: Request data dictionary
        model_name: Name of the model

    Returns:
        Dictionary with prompt_tokens and completion_tokens counts
    """
    prompt_tokens = count_tokens_in_request(request_data, model_name)
    completion_tokens = estimate_completion_tokens(request_data)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens
    }