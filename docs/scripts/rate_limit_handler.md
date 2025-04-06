# RateLimitHandler Documentation

## Overview

The `RateLimitHandler` class is a critical component of the GROQ CLOUD API client that manages rate limiting for API requests. It tracks and enforces limits on both the number of requests and tokens used within specified time periods, preventing quota overages and helping to ensure reliable API access.

## Purpose

The primary purpose of the `RateLimitHandler` class is to:

1. Track request and token usage over time
2. Enforce rate limits for requests and tokens
3. Reset counters when appropriate time periods elapse
4. Provide status information about current usage
5. Raise appropriate exceptions when limits are exceeded

## Rate Limit Concepts

The handler manages four types of rate limits:

1. **Requests per minute**: Limits the number of API requests within a one-minute period
2. **Tokens per minute**: Limits the number of tokens consumed within a one-minute period
3. **Requests per day**: Limits the number of API requests within a 24-hour period
4. **Tokens per day**: Limits the number of tokens consumed within a 24-hour period

Each limit can be set to a positive integer for a specific limit or `-1` for unlimited.

## API Reference

### Constructor

```python
def __init__(self, config: Dict[str, Any])
```

Initializes a new instance of the `RateLimitHandler` class with the specified configuration.

**Parameters:**
- `config` (Dict[str, Any]): Configuration dictionary containing rate limit settings.

**Configuration Keys:**
- `req_per_min`: Maximum requests per minute (default: 15)
- `req_per_day`: Maximum requests per day (default: 1000)
- `tokens_per_min`: Maximum tokens per minute (default: 6000)
- `tokens_per_day`: Maximum tokens per day (default: 250000)

**Example:**
```python
from CutomGroqChat.rate_limit_handler import RateLimitHandler

# Initialize with default configuration
config = {
    "req_per_min": 60,
    "req_per_day": 1000,
    "tokens_per_min": 6000,
    "tokens_per_day": 250000
}
rate_limit_handler = RateLimitHandler(config)

# Initialize with unlimited requests per minute
unlimited_config = {
    "req_per_min": -1,  # Unlimited
    "req_per_day": 1000,
    "tokens_per_min": 6000,
    "tokens_per_day": 250000
}
unlimited_handler = RateLimitHandler(unlimited_config)
```

### Methods

#### `can_make_request`

```python
def can_make_request(self, token_count: int) -> Tuple[bool, List[str]]
```

Checks if a request can be made based on the current rate limits.

**Parameters:**
- `token_count` (int): The number of tokens required for the request.

**Returns:**
- `Tuple[bool, List[str]]`: A tuple containing:
  - `bool`: True if the request can be made, False otherwise.
  - `List[str]`: A list of reasons why the request cannot be made.

**Raises:**
- `TypeError`: If token_count is not an integer.

**Example:**
```python
# Check if a request can be made
can_make, reasons = rate_limit_handler.can_make_request(100)
if can_make:
    print("Request can be made")
else:
    print(f"Request cannot be made: {', '.join(reasons)}")
```

#### `check_request`

```python
def check_request(self, token_count: int, strictly: bool = False) -> None
```

Checks if a request can be made based on the current rate limits and raises an exception if not.

**Parameters:**
- `token_count` (int): The number of tokens required for the request.
- `strictly` (bool): If True, raises RateLimitExceededException if the request can't be made.

**Raises:**
- `RateLimitExceededException`: If the request can't be made and strictly is True.
- `TypeError`: If token_count is not an integer.

**Example:**
```python
try:
    # Check if request can be made, raising an exception if not
    rate_limit_handler.check_request(100, strictly=True)
    
    # If we get here, the request can be made
    print("Request can be made")
except RateLimitExceededException as e:
    print(f"Rate limit exceeded: {e.message}")
    print(f"Limit type: {e.limit_type}")
    print(f"Time period: {e.time_period}")
```

#### `update_counters`

```python
def update_counters(self, token_count: int) -> None
```

Updates the rate limit counters after a request is made.

**Parameters:**
- `token_count` (int): The number of tokens used in the request.

**Example:**
```python
# After making a successful API request
rate_limit_handler.update_counters(150)  # Update counters with 150 tokens used
```

#### `get_status`

```python
def get_status(self) -> Dict[str, Any]
```

Gets the current status of the rate limit counters.

**Returns:**
- `Dict[str, Any]`: A dictionary containing the current status of the rate limit counters.

**Example:**
```python
# Get current rate limit status
status = rate_limit_handler.get_status()

# Print request limits
print(f"Minute requests: {status['requests']['minute']['display']}")
print(f"Daily requests: {status['requests']['day']['display']}")

# Print token limits
print(f"Minute tokens: {status['tokens']['minute']['display']}")
print(f"Daily tokens: {status['tokens']['day']['display']}")
```

### Internal Methods

#### `_reset_minute_counters`

```python
def _reset_minute_counters(self) -> None
```

Resets the minute counters and updates the last reset timestamp if a minute has passed.

#### `_reset_day_counters`

```python
def _reset_day_counters(self) -> None
```

Resets the day counters and updates the last reset timestamp if a day has passed.

#### `_reset_counters`

```python
def _reset_counters(self) -> None
```

Resets both minute and day counters if their respective time periods have passed.

## Exception Handling

The `RateLimitHandler` class uses the `RateLimitExceededException` class for error handling, which provides detailed information about the rate limit exceeded:

```python
# Example of catching RateLimitExceededException
try:
    rate_limit_handler.check_request(100, strictly=True)
    # Make the API request
except RateLimitExceededException as e:
    print(f"Error: {e.message}")
    print(f"Limit type: {e.limit_type}")
    print(f"Time period: {e.time_period}")
    print(f"Current value: {e.current_value}")
```

The `RateLimitExceededException` class includes:
- `message`: A descriptive error message
- `limit_type`: The type of limit exceeded ("request" or "token")
- `time_period`: The time period of the limit exceeded ("minute" or "day")
- `current_value`: The current value of the counter
- `limit_value`: The limit value (optional)

## Implementation Details

### Time-Based Reset

The handler automatically resets counters when their time periods elapse:
- Minute counters are reset after 60 seconds
- Day counters are reset after 86400 seconds (24 hours)

Resets are triggered when rate limit checks are performed, ensuring that counters are always up-to-date.

### Status Reporting

The `get_status` method provides a comprehensive report of current usage:

```json
{
  "requests": {
    "minute": {
      "current": 10,
      "limit": 60,
      "display": "10/60"
    },
    "day": {
      "current": 100,
      "limit": 1000,
      "display": "100/1000"
    }
  },
  "tokens": {
    "minute": {
      "current": 1000,
      "limit": 6000,
      "display": "1000/6000"
    },
    "day": {
      "current": 10000,
      "limit": 250000,
      "display": "10000/250000"
    }
  }
}
```

For unlimited limits, the "limit" field is displayed as "Unlimited".

## Integration with GROQ API Client

The `RateLimitHandler` class is designed to be integrated with the GROQ API client:

```python
from CutomGroqChat.rate_limit_handler import RateLimitHandler

# Initialize with configuration
config = {
    "req_per_min": 60,
    "req_per_day": 1000,
    "tokens_per_min": 6000,
    "tokens_per_day": 250000
}
rate_limit_handler = RateLimitHandler(config)

# Before making a request
token_count = calculate_tokens_for_request(request_data)
can_make, reasons = rate_limit_handler.can_make_request(token_count)

if can_make:
    # Make the API request
    response = make_api_request(request_data)
    
    # Update counters after successful request
    rate_limit_handler.update_counters(token_count)
else:
    print(f"Cannot make request: {', '.join(reasons)}")
```

## Best Practices

1. **Check before requesting**: Always check if a request can be made before making the actual API call
2. **Update after requesting**: Update the counters after each successful API request
3. **Monitor status**: Periodically check the status to understand current usage
4. **Handle exceptions**: Implement proper handling for `RateLimitExceededException`
5. **Set appropriate limits**: Configure rate limits based on your API quota 