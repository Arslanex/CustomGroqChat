# QueueManager Documentation

## Overview

The `QueueManager` class is a core component of the GROQ CLOUD API client that manages a priority-based asynchronous request queue system. It handles the scheduling, execution, and rate-limiting of API requests according to their specified priority levels.

## Purpose

The primary purpose of the `QueueManager` class is to:

1. Maintain three separate priority queues (high, normal, low)
2. Process requests in order of priority
3. Respect API rate limits during request processing
4. Execute requests asynchronously
5. Invoke callbacks with API responses

## API Reference

### Constructor

```python
def __init__(self, api_client: APIClient, rate_limit_handler: RateLimitHandler) -> None
```

Initializes a new instance of the `QueueManager` class with the specified API client and rate limit handler.

**Parameters:**
- `api_client` (APIClient): The API client to use for making requests.
- `rate_limit_handler` (RateLimitHandler): The handler to manage rate limits.

**Example:**
```python
from CutomGroqChat.api_client import APIClient
from CutomGroqChat.rate_limit_handler import RateLimitHandler
from CutomGroqChat.queue_manager import QueueManager

api_client = APIClient(model_config)
rate_limit_handler = RateLimitHandler(model_config)

# Initialize with API client and rate limit handler
queue_manager = QueueManager(api_client, rate_limit_handler)
```

### Methods

#### `start`

```python
def start(self) -> None
```

Starts the queue manager to process the queues.

**Example:**
```python
# Start the queue manager
queue_manager.start()
```

#### `stop`

```python
def stop(self) -> None
```

Stops the queue manager and cancels any running tasks.

**Example:**
```python
# Stop the queue manager
queue_manager.stop()
```

#### `ensure_processing`

```python
async def ensure_processing(self) -> None
```

Ensures the processing loop is running in the current event loop. This method is called automatically by `enqueue_request`.

#### `enqueue_request`

```python
async def enqueue_request(
    self,
    endpoint: str,
    payload: Dict[str, Any],
    token_count: int,
    callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    priority: str = "low"
) -> str
```

Enqueues a request to be processed according to rate limits and priority.

**Parameters:**
- `endpoint` (str): API endpoint to call.
- `payload` (Dict[str, Any]): Request payload.
- `token_count` (int): Estimated token count for the request.
- `callback` (Optional[Callable]): Optional async callback function to be called with the API response.
- `priority` (str): Priority level - "high", "normal", or "low" (default is "low").

**Returns:**
- `str`: Unique ID for this request, can be used to cancel it later.

**Raises:**
- `ValueError`: If an invalid priority level is provided.

**Example:**
```python
async def handle_response(response):
    print(f"Received response: {response}")

# Enqueue a request with normal priority
request_id = await queue_manager.enqueue_request(
    endpoint="/completions",
    payload={"prompt": "Hello, world!", "max_tokens": 100},
    token_count=10,
    callback=handle_response,
    priority="normal"
)
```

#### `get_next_request`

```python
def get_next_request(self) -> Optional[Dict[str, Any]]
```

Gets the next request to process based on priority.

**Returns:**
- Dictionary containing the next request to process, or None if no requests are available.

#### `get_queue_length`

```python
def get_queue_length(self) -> Dict[str, int]
```

Gets the length of all queues.

**Returns:**
- Dictionary with queue lengths by priority level.

**Example:**
```python
# Get queue lengths
queue_stats = queue_manager.get_queue_length()
print(f"High priority: {queue_stats['high']}")
print(f"Normal priority: {queue_stats['normal']}")
print(f"Low priority: {queue_stats['low']}")
print(f"Total: {queue_stats['total']}")
```

#### `cancel_request`

```python
async def cancel_request(self, request_id: str) -> bool
```

Cancels a pending request by ID.

**Parameters:**
- `request_id` (str): The ID of the request to cancel.

**Returns:**
- `bool`: True if the request was found and cancelled, False otherwise.

**Example:**
```python
# Cancel a request
cancelled = await queue_manager.cancel_request(request_id)
if cancelled:
    print("Request cancelled successfully")
else:
    print("Request not found or already processed")
```

#### `get_queue_status`

```python
def get_queue_status(self) -> Dict[str, Any]
```

Gets the current status of the request queue.

**Returns:**
- Dictionary with queue status information.

**Example:**
```python
# Get queue status
status = queue_manager.get_queue_status()
print(f"Queue lengths: {status['queue_lengths']}")
print(f"Running: {status['running']}")
print(f"Rate limits: {status['rate_limits']}")
```

### Internal Methods

#### `_process_queue`

```python
async def _process_queue(self) -> None
```

Processes the queue in order and handles rate limits.

#### `_process_request`

```python
async def _process_request(self, request: Dict[str, Any]) -> None
```

Processes a single request and invokes its callback.

**Parameters:**
- `request` (Dict[str, Any]): The request to process.

#### `_send_request`

```python
async def _send_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]
```

Sends a request using the API client.

**Parameters:**
- `endpoint` (str): API endpoint to call.
- `payload` (Dict[str, Any]): Request payload.

**Returns:**
- Dictionary containing the API response.

## Implementation Details

### Priority Queue System

The `QueueManager` maintains three separate queues:
1. **High Priority Queue**: For urgent requests that need immediate processing.
2. **Normal Priority Queue**: For standard priority requests.
3. **Low Priority Queue**: The default queue for non-urgent requests.

Requests are processed in strict priority order - all high priority requests are processed before any normal priority requests, and all normal priority requests are processed before any low priority requests.

### Request Processing Workflow

1. **Enqueue**: Requests are added to the appropriate queue based on priority.
2. **Process**: The processing loop checks for requests in order of priority.
3. **Rate Limit Check**: Before processing, rate limits are checked.
4. **Execution**: If rate limits allow, the request is sent to the API.
5. **Callback**: The response is passed to the callback function if provided.

### Request Map

The `QueueManager` maintains a map of request IDs to request objects for easy lookup and cancellation.

## Integration with GROQ API

The `QueueManager` class is designed to work with the GROQ API client and rate limit handler:

```python
from CutomGroqChat.api_client import APIClient
from CutomGroqChat.rate_limit_handler import RateLimitHandler
from CutomGroqChat.queue_manager import QueueManager
from CutomGroqChat.config_loader import ConfigLoader

# Load configuration
config_loader = ConfigLoader("config.json")
config = config_loader.load_config()
model_config = config_loader.get_model_config("llama-3-70b-chat")

# Initialize components
api_client = APIClient(model_config)
rate_limit_handler = RateLimitHandler(model_config)
queue_manager = QueueManager(api_client, rate_limit_handler)

# Start the queue manager
queue_manager.start()

# Use in an async context
async def send_message(message):
    return await queue_manager.enqueue_request(
        endpoint="/completions",
        payload={"prompt": message, "max_tokens": 100},
        token_count=len(message) // 4,  # Rough token estimate
        callback=handle_response,
        priority="normal"
    )
```

## Best Practices

1. **Start and Stop**: Always start the queue manager before enqueuing requests and stop it when done.
2. **Priority Levels**: Use appropriate priority levels based on the urgency of requests.
3. **Error Handling**: Implement error handling in callback functions.
4. **Token Estimation**: Provide accurate token count estimates for optimal rate limiting.
5. **Queue Monitoring**: Periodically check queue status to avoid queue buildup.

## Limitations and Known Issues

- The request processing loop runs at a fixed interval (0.1 seconds)
- Callbacks are expected to handle their own exceptions
- No built-in retry mechanism for failed requests 