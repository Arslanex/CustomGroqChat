# Token Counter Documentation

## Overview

The `token_counter` module provides utilities for accurately counting tokens in requests to the Groq Cloud API. Token counting is essential for:

1. Managing rate limits and usage quotas
2. Ensuring requests don't exceed model context windows
3. Estimating costs of API calls
4. Optimizing prompts for efficiency

The module uses the `tiktoken` library for token counting and provides functions for different types of requests (chat and completion).

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
- `messages` (List[Dict[str, str]]): List of message dictionaries
- `model_name` (str): Name of the model to use for token counting

**Returns:**
- `int`: Total number of tokens in the messages

**Example:**
```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me about token counting."}
]
token_count = count_tokens_in_messages(messages, "llama3-70b-8192")
```

### Count Tokens in Prompt

```python
def count_tokens_in_prompt(prompt: str, model_name: str) -> int
```

Counts tokens in a text prompt.

**Parameters:**
- `prompt` (str): Text prompt
- `model_name` (str): Name of the model to use for token counting

**Returns:**
- `int`: Number of tokens in the prompt

**Example:**
```python
prompt = "Generate a story about a token counter."
token_count = count_tokens_in_prompt(prompt, "llama3-70b-8192")
```

### Count Tokens in Request

```python
def count_tokens_in_request(request_data: Dict[str, Any], model_name: str) -> int
```

Counts tokens in a complete request.

**Parameters:**
- `request_data` (Dict[str, Any]): Request data dictionary
- `model_name` (str): Name of the model

**Returns:**
- `int`: Total number of tokens in the request

**Example:**
```python
request_data = {
    "model": "llama3-70b-8192",
    "messages": [
        {"role": "user", "content": "Tell me about token counting."}
    ]
}
token_count = count_tokens_in_request(request_data, "llama3-70b-8192")
```

### Estimate Completion Tokens

```python
def estimate_completion_tokens(request_data: Dict[str, Any], default_tokens: int = 100) -> int
```

Estimates the number of tokens in the completion based on request parameters.

**Parameters:**
- `request_data` (Dict[str, Any]): Request data dictionary
- `default_tokens` (int, optional): Default value if no max_tokens is specified. Defaults to 100.

**Returns:**
- `int`: Estimated number of completion tokens

**Example:**
```python
request_data = {
    "model": "llama3-70b-8192",
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
- `request_data` (Dict[str, Any]): Request data dictionary
- `model_name` (str): Name of the model

**Returns:**
- `Dict[str, int]`: Dictionary with prompt_tokens, completion_tokens, and total_tokens counts

**Example:**
```python
request_data = {
    "model": "llama3-70b-8192",
    "messages": [...],
    "max_tokens": 200
}
token_counts = count_request_and_completion_tokens(request_data, "llama3-70b-8192")
print(f"Prompt tokens: {token_counts['prompt_tokens']}")
print(f"Completion tokens: {token_counts['completion_tokens']}")
print(f"Total tokens: {token_counts['total_tokens']}")
```

## Command Line Usage

The token counter can be used from the command line:

```
python -m CutomGroqChat.token_counter [command] [arguments]
```

### Commands

#### Text

Count tokens in a text prompt:

```
python -m CutomGroqChat.token_counter text "Your text here" --model llama3-70b-8192
```

#### Chat

Count tokens in a chat conversation from a JSON file:

```
python -m CutomGroqChat.token_counter chat --file chat_messages.json --model llama3-70b-8192
```

The JSON file should contain an array of message objects.

#### Request

Count tokens in a complete request file:

```
python -m CutomGroqChat.token_counter request request_data.json --model llama3-70b-8192
```

The JSON file should contain a complete API request.

## Implementation Notes

### Default Encoding

The module uses `cl100k_base` as the default encoding, which is compatible with most modern LLMs including those used by Groq Cloud.

### Token Counting Approximations

The module makes some approximations:
- 4 tokens per message for formatting (role tokens, etc.)
- 3 tokens for overall formatting in a chat request
- 10 tokens for unknown request formats

### Function Call Handling

The module properly accounts for tokens in function calls by:
1. Converting the function call to a JSON string
2. Counting tokens in the JSON string
3. Adding those tokens to the total

## Integration with Rate Limiting

This token counter integrates well with the rate limit handler:

```python
from CutomGroqChat.token_counter import count_request_and_completion_tokens
from CutomGroqChat.rate_limit_handler import RateLimitHandler

# Initialize rate limit handler
rate_limit_handler = RateLimitHandler(config)

# Count tokens in a request
request_data = {...}
token_counts = count_request_and_completion_tokens(request_data, "llama3-70b-8192")
total_tokens = token_counts["total_tokens"]

# Check rate limits
can_make_request, reasons = rate_limit_handler.can_make_request(total_tokens)
if can_make_request:
    # Make the API request
    pass
else:
    print(f"Cannot make request: {reasons}")
``` 