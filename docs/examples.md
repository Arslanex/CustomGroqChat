# CustomGroqChat Examples

> **Note:** For detailed code-level explanations of these examples, see [Examples Explained](example/examples_explained.md).

The CustomGroqChat package includes a variety of example scripts to help you understand how to use the library effectively. These examples demonstrate different features and use cases of the API client.

## Example Files Overview

The `examples` directory contains the following example scripts:

| File | Description |
|------|-------------|
| `1_compare_all_models.py` | Runs the same prompt through all configured models for comparison |
| `2_select_model_and_chat.py` | Interactive example to select a model and chat with it |
| `3_handle_rate_limits.py` | Demonstrates basic rate limit handling with a small batch of requests |
| `3_rate_limit_test.py` | Tests exceeding rate limits by sending 35 requests at once |
| `3_rate_limit_test_aggressive.py` | Pushes rate limits further with 50 requests in batches |
| `3_rate_limit_test_exceed.py` | Simplified approach to exceed rate limits |
| `4_conversation_with_memory.py` | Shows multi-turn conversations with persistent memory |
| `5_parallel_processing.py` | Demonstrates processing multiple requests with callbacks and priorities |

## Basic Usage Examples

### Compare All Models (`examples/1_compare_all_models.py`)

This example demonstrates how to run the same prompt through all configured models and compare the results.

**Key Features:**
- Uses all available models from your configuration
- Sends the same prompt to each model
- Measures and compares response times and token usage
- Saves results to a JSON file for further analysis

**Example Usage:**
```bash
python examples/1_compare_all_models.py
```

**What You'll Learn:**
- How to get a list of available models
- How to send requests to different models
- How to measure performance metrics
- How to handle and compare model responses

### Select Model and Chat (`examples/2_select_model_and_chat.py`)

This interactive example lets you select a model from your configuration and chat with it.

**Key Features:**
- Lists all available models from your configuration
- Allows interactive model selection
- Takes user input for prompts
- Displays complete responses with token usage information

**Example Usage:**
```bash
python examples/2_select_model_and_chat.py
```

**What You'll Learn:**
- How to present model options to users
- How to handle user input safely
- How to format and send custom messages
- How to display and interpret token usage

## Rate Limiting Examples

### Handle Rate Limits (`examples/3_handle_rate_limits.py`)

This example demonstrates how the client handles basic rate limiting with a small batch of requests.

**Key Features:**
- Sends a manageable number of requests (8) simultaneously
- Shows how to check rate limit status before and after
- Uses `asyncio.as_completed()` to process responses as they arrive
- Includes basic error handling

**Example Usage:**
```bash
python examples/3_handle_rate_limits.py
```

**What You'll Learn:**
- How to check current rate limit usage
- How to send multiple requests concurrently
- How to process responses as they complete
- How CustomGroqChat automatically manages rate limits

### Rate Limit Test (`examples/3_rate_limit_test.py`)

This example intentionally exceeds the rate limit by sending 35 requests in parallel (beyond the typical 30 per minute limit).

**Key Features:**
- Attempts to exceed the per-minute rate limit
- Shows how CustomGroqChat manages the queue
- Demonstrates automatic request distribution across time windows
- Provides detailed monitoring of request status

**Example Usage:**
```bash
python examples/3_rate_limit_test.py
```

**Test Results:**
- All 35 requests complete successfully despite exceeding the rate limit
- Total execution time is approximately 63 seconds
- The library automatically distributes requests across multiple rate limit windows
- No rate limit errors are encountered

### Aggressive Rate Limit Test (`examples/3_rate_limit_test_aggressive.py`)

This example pushes rate limits even further by sending 50 requests in batches.

**Key Features:**
- Sends requests in two batches (30, then 20 more)
- Checks rate limit status between batches
- Uses a timeout to prevent indefinite waiting
- Processes any completed responses regardless of errors

**Example Usage:**
```bash
python examples/3_rate_limit_test_aggressive.py
```

**Test Results:**
- During testing, this script triggers rate limit errors after around 27 requests
- The client detects approaching rate limits and begins throttling requests
- Some rate limit error messages may be observed
- Confirms the library can detect when it's approaching a true rate limit

### Simple Rate Limit Exceed Test (`examples/3_rate_limit_test_exceed.py`)

This example uses a simplified approach to exceed rate limits with a specific model.

**Key Features:**
- Directly specifies the model name ("llama-3.1-8b-instant")
- Sends 35 simple math questions at once
- Uses `asyncio.gather()` to wait for all responses
- Provides a clean, straightforward test case

**Example Usage:**
```bash
python examples/3_rate_limit_test_exceed.py
```

**Test Results:**
- All 35 requests complete successfully despite exceeding the rate limit
- Total execution time is approximately 62 seconds
- The queue system automatically paces the requests to fit within rate limits
- No errors are encountered

## Advanced Examples

### Conversation with Memory (`examples/4_conversation_with_memory.py`)

This example demonstrates how to maintain a conversation with persistent memory across sessions.

**Key Features:**
- Saves and loads conversation history from JSON files
- Allows picking up previous conversations
- Maintains context across multiple exchanges
- Shows proper message history management

**Example Usage:**
```bash
python examples/4_conversation_with_memory.py
```

**What You'll Learn:**
- How to structure a conversation manager class
- How to maintain conversation context
- How to persist conversations to disk
- How to implement a conversational UI

### Parallel Processing (`examples/5_parallel_processing.py`)

This example demonstrates how to process multiple requests in parallel with different priorities and callbacks.

**Key Features:**
- Creates a mix of high, medium, and low priority requests
- Uses callbacks to process responses asynchronously
- Shows how priority affects processing order
- Includes detailed timing and monitoring

**Example Usage:**
```bash
python examples/5_parallel_processing.py
```

**What You'll Learn:**
- How to use priority levels to control request order
- How to implement and use callbacks for async processing
- How to manage multiple concurrent requests of varying importance
- How to monitor queue status during processing

## Using the Example Configuration

The examples directory includes a `config.json` file that you can use as a template for your own configuration. This file contains sample model configurations with appropriate rate limits.

**Example Configuration:**
```json
{
  "llama-3.1-8b-instant": {
    "base_url": "https://api.groq.com/openai/v1",
    "api_key": "YOUR_GROQ_API_KEY",
    "req_per_minute": 30,
    "req_per_day": 1200,
    "token_per_minute": 7500,
    "token_per_day": 300000,
    "context_window": 8192
  },
  "gemma2-9b-it": {
    "base_url": "https://api.groq.com/openai/v1",
    "api_key": "YOUR_GROQ_API_KEY",
    "req_per_minute": 10,
    "req_per_day": 700,
    "token_per_minute": 4500,
    "token_per_day": 200000,
    "context_window": 32768
  }
}
```

## Key Takeaways from the Examples

After exploring these examples, here are the key insights about CustomGroqChat:

1. **Intelligent Queue Management**: CustomGroqChat automatically manages request queues to prevent rate limit errors by pacing requests appropriately.

2. **Automatic Request Distribution**: Requests exceeding the per-minute limit are automatically spread across multiple minutes, allowing all requests to complete successfully.

3. **Priority System**: The library's priority queue effectively ensures that critical requests are processed before less important ones.

4. **Persistent Conversations**: The conversation memory system makes it easy to build chatbots with context awareness and persistence.

5. **Asynchronous Processing**: CustomGroqChat's async design makes it efficient for handling many concurrent requests.

6. **Multiple Model Support**: The library seamlessly works with different GROQ models, each with their own rate limits and capabilities.

7. **Callback Functionality**: The callback system allows for efficient, non-blocking processing of responses.

## Running the Examples

To run any of these examples:

1. Copy the `config.json` file from the examples directory to your working directory
2. Update it with your GROQ API key
3. Install the CustomGroqChat package and dependencies
4. Run the example script with Python

```bash
# Install dependencies
pip install customgroqchat

# Copy and edit the config file
cp examples/config.json ./config.json
# Edit config.json with your API key

# Run an example
python examples/1_compare_all_models.py 