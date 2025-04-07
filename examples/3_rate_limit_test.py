"""
Rate Limit Test - Modified from Example 3

This script intentionally exceeds the configured rate limits to
demonstrate how the error handling system works.
"""
import asyncio
import time
import random
from CustomGroqChat import GroqClient, RateLimitExceededException, CustomGroqChatException

# Quick questions to send in rapid succession to trigger rate limits
QUESTIONS = [
    "What is 1+1?",
    "What is your name?",
    "What is today?",
    "Hello?",
    "What is red?",
    "Who are you?",
    "Tell me a joke.",
    "What is 2+2?",
    "What is Python?",
    "What is an API?",
    "What is the sun?",
    "What is a dog?",
    "What is water?",
    "How are you?",
    "What is the time?",
    "What is a car?",
    "What is music?",
    "What is a book?",
    "What is food?",
    "What is sleep?",
    "What is a computer?",
    "What is code?",
    "What is language?",
    "What is a tree?",
    "What is a number?",
    "What is a question?",
    "What is a word?",
    "What is a sentence?",
    "What is a paragraph?",
    "What is a document?",
    "What is an essay?",
    "What is a novel?",
    "What is a story?",
    "What is art?",
    "What is science?",
    "What is math?",
    "What is history?",
    "What is geography?",
    "What is physics?",
    "What is chemistry?"
]

async def send_with_retry(client, model, message, max_retries=3, base_delay=2):
    """Send a request with retry logic for rate limiting."""
    retries = 0
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Keep answers very short and simple."},
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
                print(f"❌ Exceeded maximum retries ({max_retries}) for message: {message}")
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
        
        # Use the llama model specifically
        model = "llama-3.1-8b-instant"  # This has 30 requests per minute limit
        if model not in models:
            print(f"Llama model not found, using alternative: {models[0]}")
            model = models[0]
            
        print(f"Using model: {model}")
        
        # Fixed rate limit of 30 per minute as specified by user
        req_per_minute = 30
        print(f"Rate limit: {req_per_minute} requests per minute")
        
        # We'll attempt to send more requests than the limit allows
        num_requests = req_per_minute + 5  # 5 more than the limit
        print(f"Will attempt to send {num_requests} requests (exceeding the limit of {req_per_minute})")
        
        # Get queue status before we start
        status = await client.get_queue_status()
        print("\nInitial queue status:")
        print(f"  Requests: Minute {status['rate_limits']['requests']['minute']['display']}, "
              f"Day {status['rate_limits']['requests']['day']['display']}")
        
        # Process multiple requests in parallel to trigger rate limits
        print("\nSending multiple requests in parallel to trigger rate limits...")
        start_time = time.time()
        
        # Process a batch of questions
        tasks = []
        for i, question in enumerate(QUESTIONS[:num_requests]):
            print(f"Queueing request {i+1}: {question}")
            
            # No delay between requests to ensure we hit the rate limit
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
                print(f"  A: {answer[:50]}..." if len(answer) > 50 else f"  A: {answer}")
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