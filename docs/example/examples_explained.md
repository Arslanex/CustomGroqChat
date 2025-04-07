# CustomGroqChat Examples Explained

This document provides detailed code-level explanations of the key example scripts in the CustomGroqChat package. It's designed to help you understand the important patterns and techniques used in each example.

## 1. Compare All Models

**File:** `examples/1_compare_all_models.py`

This example demonstrates running the same prompt through multiple models and comparing their performance.

### Key Code Sections

#### Getting Available Models
```python
# Get all available models
models = client.get_available_models()
if not models:
    print("No models found in configuration!")
    return
    
print(f"Found {len(models)} models: {', '.join(models)}")
```
This code retrieves the list of models from your configuration file, making it easy to run tests across all configured models.

#### Sending the Same Prompt to Multiple Models
```python
# Process each model sequentially
for model in models:
    print(f"\nQuerying model: {model}")
    start_time = time.time()
    
    try:
        response = await client.chat_completion(
            model_name=model,
            messages=messages
        )
        
        # Extract the response content
        content = response["choices"][0]["message"]["content"]
        token_usage = response["usage"]
        
        # Calculate time taken
        time_taken = time.time() - start_time
        
        # Store results
        results[model] = {
            "content": content,
            "token_usage": token_usage,
            "time_taken": time_taken
        }
        
        print(f"Response received in {time_taken:.2f} seconds")
        print(f"Used {token_usage['total_tokens']} tokens")
        
    except CustomGroqChatException as e:
        print(f"Error with model {model}: {e}")
        results[model] = {"error": str(e)}
```
This loop processes each model sequentially, measuring performance metrics and handling errors gracefully.

#### Saving Results for Analysis
```python
# Save results to a file
with open("model_comparison_results.json", "w") as f:
    json.dump(results, f, indent=2)
    
print("\nResults saved to model_comparison_results.json")
```
The results are saved to a JSON file, making it easy to analyze and compare model performance later.

## 2. Select Model and Chat

**File:** `examples/2_select_model_and_chat.py`

This example provides an interactive interface for model selection and chatting.

### Key Code Sections

#### Interactive Model Selection
```python
# Let the user select a model
print("\nAvailable models:")
for i, model in enumerate(models, 1):
    print(f"{i}. {model}")
    
# Get user selection
selected_idx = 0
while selected_idx < 1 or selected_idx > len(models):
    try:
        selected_idx = int(input(f"\nSelect a model (1-{len(models)}): "))
        if selected_idx < 1 or selected_idx > len(models):
            print("Invalid selection. Please try again.")
    except ValueError:
        print("Please enter a number.")

# Get the selected model
selected_model = models[selected_idx - 1]
print(f"\nYou selected: {selected_model}")
```
This code creates a numbered list of models and safely handles user input for selection.

#### Getting User Input and Creating Messages
```python
# Get the user's message
user_message = input("\nEnter your message to the model: ")

# Create the messages array
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": user_message}
]
```
The code gets user input and formats it as part of a proper messages array with a system prompt.

#### Displaying Token Usage
```python
# Show token usage
usage = response["usage"]
print(f"\nToken usage: {usage['prompt_tokens']} prompt + {usage['completion_tokens']} completion = {usage['total_tokens']} total")
```
This section provides a detailed breakdown of token usage, which is helpful for understanding costs and context window usage.

## 3. Handle Rate Limits

**File:** `examples/3_handle_rate_limits.py`

This example demonstrates the basic rate limit handling capabilities of CustomGroqChat.

### Key Code Sections

#### Checking Rate Limits
```python
# Check initial limits
limits = await client.check_model_limits(model_name)
print(f"Initial limits - minute: {limits['minute_used']}/{limits['minute_limit']}, day: {limits['day_used']}/{limits['day_limit']}")
```
This code checks and displays the current rate limit status before making any requests.

#### Sending Multiple Requests
```python
# Send 8 requests with retry
start_time = time.time()
tasks = []

for i in range(1, 9):
    messages = [
        {"role": "user", "content": f"What is {i}+{i}?"}
    ]
    
    task = client.chat_completion(
        model_name=model_name,
        messages=messages
    )
    tasks.append(task)
```
Here, multiple simple request tasks are created and added to a list.

#### Processing Responses as They Complete
```python
# Wait for all to complete with retry
responses = []
for task in asyncio.as_completed(tasks):
    try:
        response = await task
        responses.append(response)
        print(f"Request completed: {response['choices'][0]['message']['content']}")
    except Exception as e:
        print(f"Request failed: {e}")
```
The `asyncio.as_completed()` function allows processing responses as soon as they're ready, rather than waiting for all to complete.

## 4. Conversation with Memory

**File:** `examples/4_conversation_with_memory.py`

This example demonstrates maintaining conversation history across multiple exchanges.

### Key Code Sections

#### Conversation Manager Class
```python
class ConversationManager:
    def __init__(self, client, model_name, conversation_id=None):
        self.client = client
        self.model_name = model_name
        self.conversation_id = conversation_id or f"conversation_{int(time.time())}"
        self.messages = []
        self.load_conversation()
```
The `ConversationManager` class encapsulates all conversation-related functionality, making it reusable and maintainable.

#### Loading Existing Conversations
```python
def load_conversation(self):
    # Create conversations directory if it doesn't exist
    os.makedirs("conversations", exist_ok=True)
    
    # Try to load existing conversation
    conversation_path = f"conversations/{self.conversation_id}.json"
    if os.path.exists(conversation_path):
        try:
            with open(conversation_path, "r") as f:
                data = json.load(f)
                self.messages = data.get("messages", [])
                print(f"Loaded conversation with {len(self.messages)} messages")
        except Exception as e:
            print(f"Error loading conversation: {e}")
            self.messages = []
    
    # Initialize with system message if empty
    if not self.messages:
        self.messages = [
            {"role": "system", "content": "You are a helpful assistant with a memory of the conversation history."}
        ]
        self.save_conversation()
```
This method loads an existing conversation from disk or initializes a new one with a system message.

#### Conversation Loop
```python
# Main conversation loop
while True:
    user_input = input("\nYou (or 'exit' to quit): ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        break
        
    # Add user message and get response
    await conversation_manager.add_user_message(user_input)
    print("Assistant is thinking...")
    response = await conversation_manager.get_response()
    
    # Print the response
    print(f"\nAssistant: {response['content']}")
    print(f"(Used {response['usage'].get('total_tokens', 'N/A')} tokens)")
```
The main loop adds user messages, gets responses, and displays them with token usage information.

## 5. Parallel Processing

**File:** `examples/5_parallel_processing.py`

This example demonstrates processing multiple requests with different priorities and callbacks.

### Key Code Sections

#### Callback Function
```python
async def process_response(response, metadata):
    """Process a response asynchronously"""
    priority = metadata.get("priority", "unknown")
    index = metadata.get("index", 0)
    content = response["choices"][0]["message"]["content"]
    tokens = response["usage"]["total_tokens"]
    elapsed = time.time() - metadata["start_time"]
    
    print(f"[{elapsed:.2f}s] Completed request {index} (priority: {priority}): {content[:50]}... ({tokens} tokens)")
```
This async callback function processes responses as they complete, showing how to handle asynchronous responses with metadata.

#### Setting Request Priorities
```python
# High priority task (urgent question)
print("Queuing 1 high priority request...")
messages = [{"role": "user", "content": "URGENT: What is the current time in New York?"}]

high_task = client.chat_completion(
    model_name=model_name,
    messages=messages,
    priority=1,  # High priority
    callback=process_response,
    callback_metadata={
        "priority": "high",
        "index": 1,
        "start_time": start_time
    }
)
tasks.append(high_task)
```
This code sets a high priority (1) for an urgent request, which will be processed before other lower-priority requests.

#### Using Callbacks with Metadata
```python
task = client.chat_completion(
    model_name=model_name,
    messages=messages,
    priority=10,  # Low priority
    callback=process_response,
    callback_metadata={
        "priority": "low",
        "index": i,
        "start_time": start_time
    }
)
```
The `callback` and `callback_metadata` parameters allow asynchronous processing of responses when they're ready, without blocking the main flow.

## Key Design Patterns

Across these examples, several key design patterns emerge:

1. **Proper Resource Management**
   ```python
   try:
       # Use the client
   finally:
       # Always close the client
       await client.close()
   ```
   Resources are properly initialized and closed using try/finally blocks.

2. **Error Handling**
   ```python
   try:
       response = await task
   except Exception as e:
       print(f"Request failed: {e}")
   ```
   Robust error handling ensures that failures in one request don't crash the entire application.

3. **Asynchronous Patterns**
   ```python
   # Create tasks
   tasks = []
   for i in range(1, 6):
       task = client.chat_completion(...)
       tasks.append(task)
   
   # Wait for all to complete
   responses = await asyncio.gather(*tasks)
   ```
   Proper async patterns enable concurrent processing for better performance.

4. **Encapsulation**
   ```python
   class ConversationManager:
       # Encapsulates all conversation logic
   ```
   Complex functionality is encapsulated in classes for better organization and reuse.

5. **Configuration Management**
   ```python
   # Get available models
   models = client.get_available_models()
   ```
   Configuration is loaded once and accessed through clean APIs.

## Running the Examples

To run any of these examples:

1. Ensure you have the CustomGroqChat package installed
2. Create a `config.json` file with your API key and model configurations
3. Run the example with Python 3.7 or later

```bash
python examples/4_conversation_with_memory.py
```

Each example is designed to be educational and practical, demonstrating real-world usage patterns for the CustomGroqChat library. 