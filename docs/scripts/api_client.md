# API Client Documentation

## Overview

The `api_client` module provides a robust, asynchronous HTTP client for interacting with the Groq Cloud API. It handles authentication, request formatting, response parsing, and error management, providing a clean interface for making API calls.

## Key Features

- **Asynchronous API**: Uses `aiohttp` for non-blocking API calls
- **Connection pooling**: Manages HTTP sessions for optimal performance
- **Error handling**: Provides detailed error information via exceptions
- **JSON processing**: Automatically handles JSON serialization and deserialization
- **Authentication**: Manages API key authentication

## Classes

### APIClient

The main client class for making requests to the Groq API.

```python
from CutomGroqChat.api_client import APIClient

# Initialize the client
client = APIClient(
    base_url="https://api.groq.com",
    api_key="your-api-key"
)

# Make a request
response = await client.post_request(
    endpoint="chat/completions",
    payload={
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": "Hello, world!"}]
    }
)

# Close the client when done
await client.close()
```

#### Constructor Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `base_url` | `str` | The base URL for the API (e.g., "https://api.groq.com") |
| `api_key` | `str` | The API key for authentication |

#### Methods

##### `async post_request(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]`

Makes a POST request to the specified API endpoint.

**Parameters:**
- `endpoint` (str): The API endpoint to make the request to (e.g., "chat/completions")
- `payload` (Dict[str, Any]): The payload to send in the request

**Returns:**
- Dict[str, Any]: The parsed JSON response from the API

**Raises:**
- `APICallException`: If there is an error making the API call

**Example:**
```python
response = await client.post_request(
    endpoint="chat/completions",
    payload={
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me about Groq."}
        ]
    }
)
```

##### `async close() -> None`

Closes the aiohttp session if it exists.

**Example:**
```python
await client.close()
```

#### Protected Methods

##### `async _get_session() -> aiohttp.ClientSession`

Gets an aiohttp session. If one does not exist, creates it.

**Returns:**
- `aiohttp.ClientSession`: The aiohttp session.

## Error Handling

The API client uses the `APICallException` class to provide detailed error information when API calls fail.

### Exception Types

#### `APICallException`

Raised when an API call fails for any reason.

**Attributes:**
- `message` (str): A descriptive error message
- `status_code` (int): The HTTP status code, if available

**Error Scenarios:**
- API errors (4xx, 5xx status codes)
- JSON parsing errors
- Network errors

## Best Practices

### Session Management

The API client manages HTTP sessions internally to optimize performance. For best results:

1. **Create a single client** for multiple requests
2. **Close the client** when finished to release resources
3. **Use async context managers** when appropriate

```python
async with APIClient(base_url, api_key) as client:
    response = await client.post_request(endpoint, payload)
```

### Error Handling

Use try/except blocks to handle API errors gracefully:

```python
try:
    response = await client.post_request(endpoint, payload)
    # Process response
except APICallException as e:
    if e.status_code == 429:
        # Handle rate limiting
        print(f"Rate limited: {e.message}")
    elif e.status_code >= 500:
        # Handle server errors
        print(f"Server error: {e.message}")
    else:
        # Handle other errors
        print(f"API error: {e.message}")
```

## Dependencies

The API client has the following dependencies:

- **aiohttp**: For asynchronous HTTP requests
- **json**: For JSON serialization and deserialization
- **typing**: For type annotations

## Implementation Details

### Headers

The client automatically sets the following headers:

```python
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
```

### Request Flow

1. Get or create an HTTP session
2. Construct the full URL from the base URL and endpoint
3. Send the POST request with the JSON payload
4. Parse the response as JSON
5. Check for error status codes and raise exceptions if needed
6. Return the parsed response

## Integration with Rate Limiting

This API client is designed to work with the rate limiting system. When rate limits are exceeded, the API may return 429 (Too Many Requests) status codes, which will be exposed as `APICallException` with status code 429. 