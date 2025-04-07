# Token Counter Documentation

## Overview

The `token_counter.py` module provides utilities for accurately counting tokens in requests to the Groq Cloud API. This module is used by the [`RequestHandler`](request_handler.md) to validate token limits and by the [`RateLimitHandler`](rate_limit_handler.md) to track token usage.

Token counting is essential for:

1. Managing rate limits and usage quotas
2. Ensuring requests don't exceed model context windows
3. Estimating costs of API calls
4. Optimizing prompts for efficiency

The module uses the `tiktoken` library for token counting and provides functions for different types of requests (chat and completion). These functions are also exported directly by the package - see [Package Exports](package_exports.md) for details.

## Token Counting Concepts

Tokens are the basic units of text that language models process. In most modern tokenizers:
- Words are usually broken into multiple tokens
- Common words or phrases may be a single token
- Special characters, spaces, and formatting can also be tokens

For Groq Cloud API, token counting is important for:
- Rate limiting (tokens per minute/day)
- Context window management (ensuring requests fit within the model's limits)
- Cost calculation (API usage is typically billed per token)

## API Reference

### Count Tokens in Message

```python
def count_tokens_in_message(message: Dict[str, str], encoding: tiktoken.Encoding) -> int
```

Counts tokens in a single chat message.

**Parameters:**
- `message` (Dict[str, str]): A message dictionary with 'role' and 'content' keys
- `encoding` (tiktoken.Encoding): The tokenizer encoding to use

**Returns:**
- `int`: Number of tokens in the message

**Example:**
```python
message = {"role": "user", "content": "Tell me about token counting."}
encoding = tiktoken.get_encoding("cl100k_base")
token_count = count_tokens_in_message(message, encoding)
```

### Count Tokens in Messages

```python
def count_tokens_in_messages(messages: List[Dict[str, str]], model_name: str) -> int
```

Counts tokens in a list of chat messages.

**Parameters:**
- `messages` (List[Dict[str, str]]): List of message dictionaries with 'role' and 'content' keys
- `model_name` (str): Name of the model to use for token counting (e.g., "llama-3.1-8b-instant")

**Returns:**
- `int`: Total number of tokens in the messages

**Example:**
```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me about token counting."}
]
token_count = count_tokens_in_messages(messages, "llama-3.1-8b-instant")
```

### Count Tokens in Prompt

```python
def count_tokens_in_prompt(prompt: str, model_name: str) -> int
```

Counts tokens in a text prompt.

**Parameters:**
- `prompt` (str): Text prompt string
- `model_name` (str): Name of the model to use for token counting (e.g., "llama-3.1-8b-instant")

**Returns:**
- `int`: Number of tokens in the prompt

**Example:**
```python
prompt = "Generate a story about a token counter."
token_count = count_tokens_in_prompt(prompt, "llama-3.1-8b-instant")
```

### Count Tokens in Request

```python
def count_tokens_in_request(request_data: Dict[str, Any], model_name: str) -> int
```

Counts tokens in a complete request payload.

**Parameters:**
- `request_data` (Dict[str, Any]): Request data dictionary containing either 'messages' or 'prompt'
- `model_name` (str): Name of the model to use for token counting

**Returns:**
- `int`: Total number of tokens in the request

**Example:**
```python
request_data = {
    "model": "llama-3.1-8b-instant",
    "messages": [
        {"role": "user", "content": "Tell me about token counting."}
    ]
}
token_count = count_tokens_in_request(request_data, "llama-3.1-8b-instant")
```

### Estimate Completion Tokens

```python
def estimate_completion_tokens(request_data: Dict[str, Any], default_tokens: int = 100) -> int
```

Estimates the number of tokens in the completion based on request parameters.

**Parameters:**
- `request_data` (Dict[str, Any]): Request data dictionary that may contain 'max_tokens'
- `default_tokens` (int, optional): Default value if no max_tokens is specified. Defaults to 100.

**Returns:**
- `int`: Estimated number of completion tokens

**Example:**
```python
request_data = {
    "model": "llama-3.1-8b-instant",
    "messages": [...],
    "max_tokens": 200
}
token_count = estimate_completion_tokens(request_data)
```

### Count Request and Completion Tokens

```python
def count_request_and_completion_tokens(request_data: Dict[str, Any], model_name: str) -> Dict[str, int]
```

Counts tokens for both the request and the estimated completion.

**Parameters:**
- `request_data` (Dict[str, Any]): Request data dictionary containing message or prompt data
- `model_name` (str): Name of the model to use for token counting

**Returns:**
- `Dict[str, int]`: Dictionary with 'prompt_tokens', 'completion_tokens', and 'total_tokens' counts

**Example:**
```python
request_data = {
    "model": "llama-3.1-8b-instant",
    "messages": [...],
    "max_tokens": 200
}
token_counts = count_request_and_completion_tokens(request_data, "llama-3.1-8b-instant")
print(f"Prompt tokens: {token_counts['prompt_tokens']}")
print(f"Completion tokens: {token_counts['completion_tokens']}")
print(f"Total tokens: {token_counts['total_tokens']}")
```

## Command Line Usage

The token counter can be used from the command line:

```
python -m CustomGroqChat.token_counter [command] [arguments]
```

### Commands

#### Text

Count tokens in a text prompt:

```
python -m CustomGroqChat.token_counter text "Your text here" --model llama-3.1-8b-instant
```

#### Chat

Count tokens in a chat conversation from a JSON file:

```
python -m CustomGroqChat.token_counter chat --file chat_messages.json --model llama-3.1-8b-instant
```

The JSON file should contain an array of message objects.

## Related Documentation

- [RequestHandler](request_handler.md) - Uses the token counter to validate requests
- [RateLimitHandler](rate_limit_handler.md) - Uses the token counter to track usage
- [Package Exports](package_exports.md) - Information about token counting functions in the public API
- [Advanced Usage Guide](../usage%20guide/advanced_usage.md#token-counting-and-budget-management) - Guide to token counting in applications 