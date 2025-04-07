"""
Example 3: Handle Rate Limits

This script demonstrates how to handle rate limit errors by sending multiple requests
in quick succession and implementing a retry mechanism with exponential backoff.
"""
import asyncio
import time
import random
from CustomGroqChat import GroqClient, RateLimitExceededException, CustomGroqChatException

# Sample questions to send in rapid succession
SAMPLE_QUESTIONS = [
    "What is the capital of France?",
    "How does photosynthesis work?",
    "Explain the theory of relativity simply.",
    "What are the main features of Python?",
    "How do neural networks learn?",
    "What is the difference between HTTP and HTTPS?",
    "Explain quantum computing in simple terms.",
    "What causes climate change?",
    "How does blockchain technology work?",
    "What are the benefits of regular exercise?",
    "How does DNA store genetic information?",
    "What is the process of machine learning?",
    "Explain the water cycle.",
    "What is artificial intelligence?",
    "How do vaccines work?",
]

async def send_with_retry(client, model, message, max_retries=3, base_delay=2):
    """Send a request with retry logic for rate limiting."""
    retries = 0
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": message}
    ]
    
    while True:
        try:
            # Try to send the request
            response = await client.chat_completion(
                model_name=model,
                messages=messages,
                temperature=0.7
            )
            
            # If successful, return the response
            return response
            
        except RateLimitExceededException as e:
            # Handle rate limit specifically
            retries += 1
            if retries > max_retries:
                print(f"❌ Exceeded maximum retries ({max_retries}) for message: {message[:30]}...")
                raise
                
            # Calculate backoff with jitter (randomization)
            delay = base_delay ** retries + random.uniform(0, 1)
            print(f"⚠️ Rate limit exceeded. Retry {retries}/{max_retries} after {delay:.2f}s delay. ({e.limit_type} limit)")
            await asyncio.sleep(delay)
            
        except CustomGroqChatException as e:
            # Handle other CustomGroqChat exceptions
            print(f"❌ Error: {e}")
            raise

async def test_rate_limits():
    # Initialize the client
    print("Initializing client...")
    client = GroqClient()
    await client.initialize()
    
    try:
        # Get available models
        models = client.get_available_models()
        if not models:
            print("No models found in configuration!")
            return
            
        # Select the first model for simplicity, or let user select
        model = models[0]
        print(f"Using model: {model}")
        
        # Get queue status before we start
        status = await client.get_queue_status()
        print("\nInitial queue status:")
        print(f"  Requests: Minute {status['rate_limits']['requests']['minute']['display']}, "
              f"Day {status['rate_limits']['requests']['day']['display']}")
        print(f"  Tokens: Minute {status['rate_limits']['tokens']['minute']['display']}, "
              f"Day {status['rate_limits']['tokens']['day']['display']}")
        
        # Process multiple requests in parallel to trigger rate limits
        print("\nSending multiple requests in parallel to test rate limiting...")
        start_time = time.time()
        
        # Process a batch of questions
        tasks = []
        for i, question in enumerate(SAMPLE_QUESTIONS[:8]):  # Use first 8 questions
            print(f"Queueing request {i+1}: {question[:30]}...")
            # Small delay to make log output more readable
            await asyncio.sleep(0.1)
            
            # Create task for each request
            task = asyncio.create_task(
                send_with_retry(client, model, question)
            )
            tasks.append((i+1, question, task))
        
        # Wait for all tasks to complete
        print("\nWaiting for all requests to complete...")
        completed = 0
        failed = 0
        
        for i, question, task in tasks:
            try:
                response = await task
                answer = response["choices"][0]["message"]["content"]
                tokens = response.get("usage", {}).get("total_tokens", "N/A")
                print(f"\n✅ Request {i} completed:")
                print(f"  Q: {question}")
                print(f"  A: {answer[:100]}..." if len(answer) > 100 else f"  A: {answer}")
                print(f"  Tokens: {tokens}")
                completed += 1
                
            except Exception as e:
                print(f"\n❌ Request {i} failed: {e}")
                failed += 1
        
        # Get queue status after we're done
        status = await client.get_queue_status()
        print("\nFinal queue status:")
        print(f"  Requests: Minute {status['rate_limits']['requests']['minute']['display']}, "
              f"Day {status['rate_limits']['requests']['day']['display']}")
        print(f"  Tokens: Minute {status['rate_limits']['tokens']['minute']['display']}, "
              f"Day {status['rate_limits']['tokens']['day']['display']}")
        
        # Print summary
        elapsed = time.time() - start_time
        print(f"\nTest completed in {elapsed:.2f} seconds.")
        print(f"Successful requests: {completed}")
        print(f"Failed requests: {failed}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up
        await client.close()
        print("\nClient closed.")

if __name__ == "__main__":
    asyncio.run(test_rate_limits()) 