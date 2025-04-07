"""
Rate Limit Test (Aggressive Version)

This script aggressively tries to exceed the rate limits by sending a large number
of requests in rapid succession to demonstrate how the error handling works.
"""
import asyncio
import time
import random
from CustomGroqChat import GroqClient, RateLimitExceededException, CustomGroqChatException

# Simple questions to send in rapid succession
QUESTIONS = [
    "What is 1+1?",
    "What is 2+2?",
    "What is 3+3?",
    "What is 4+4?",
    "What is 5+5?",
    "What is 6+6?",
    "What is 7+7?",
    "What is 8+8?",
    "What is 9+9?",
    "What is 10+10?",
    "What is red?",
    "What is blue?",
    "What is green?",
    "What is yellow?",
    "What is purple?",
    "What is black?",
    "What is white?",
    "What is gray?",
    "What is brown?",
    "What is orange?",
] * 5  # Repeat the list 5 times to have 100 questions

async def send_request(client, model, message, request_id, with_retry=False, max_retries=2, base_delay=1):
    """Send a request with optional retry logic."""
    retries = 0
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Keep answers to one sentence."},
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
            return response, request_id, None
            
        except RateLimitExceededException as e:
            error_details = f"Rate limit exceeded: {e.limit_type} limit"
            print(f"⚠️ Request {request_id}: {error_details}")
            
            if with_retry and retries < max_retries:
                retries += 1
                # Calculate backoff with jitter (randomization)
                delay = base_delay ** retries + random.uniform(0, 0.5)
                print(f"   Retry {retries}/{max_retries} after {delay:.2f}s delay.")
                await asyncio.sleep(delay)
            else:
                # Return the error
                return None, request_id, error_details
                
        except CustomGroqChatException as e:
            error_details = f"Error: {e}"
            print(f"❌ Request {request_id}: {error_details}")
            return None, request_id, error_details

async def test_rate_limits():
    """Run the aggressive rate limit test."""
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
        
        # Use the llama model specifically (high req/min limit)
        model = "llama-3.1-8b-instant"
        if model not in models:
            print(f"Llama model not found, using alternative: {models[0]}")
            model = models[0]
            
        print(f"Using model: {model}")
        
        # Configurable test parameters
        total_requests = 50  # Total number of requests to send
        batch_size = 30     # How many requests to send simultaneously
        enable_retry = True # Whether to retry failed requests
        
        print(f"Test parameters:")
        print(f"  Total requests: {total_requests}")
        print(f"  Batch size: {batch_size}")
        print(f"  Retry enabled: {enable_retry}")
        
        # Get queue status before we start
        status = await client.get_queue_status()
        print("\nInitial queue status:")
        print(f"  Requests: Minute {status['rate_limits']['requests']['minute']['display']}, "
              f"Day {status['rate_limits']['requests']['day']['display']}")
        
        # Start the test
        print("\nSending requests in batches to trigger rate limits...")
        start_time = time.time()
        
        # Results tracking
        results = {
            "completed": 0,
            "failed": 0,
            "rate_limited": 0,
            "other_errors": 0,
            "responses": []
        }
        
        # Send requests in batches
        for batch_start in range(0, total_requests, batch_size):
            batch_end = min(batch_start + batch_size, total_requests)
            batch_size_actual = batch_end - batch_start
            
            print(f"\nStarting batch {batch_start//batch_size + 1} ({batch_size_actual} requests)...")
            batch_start_time = time.time()
            
            # Create tasks for the batch
            tasks = []
            for i in range(batch_start, batch_end):
                question = QUESTIONS[i % len(QUESTIONS)]
                request_id = i + 1
                print(f"  Queueing request {request_id}: {question}")
                
                task = asyncio.create_task(
                    send_request(
                        client=client,
                        model=model,
                        message=question,
                        request_id=request_id,
                        with_retry=enable_retry
                    )
                )
                tasks.append(task)
            
            # Wait for all tasks in the batch to complete
            batch_results = await asyncio.gather(*tasks)
            
            # Process the results
            for response, request_id, error in batch_results:
                if response:
                    # Successful request
                    answer = response["choices"][0]["message"]["content"]
                    tokens = response.get("usage", {}).get("total_tokens", "N/A")
                    results["completed"] += 1
                    results["responses"].append({
                        "request_id": request_id,
                        "question": QUESTIONS[(request_id-1) % len(QUESTIONS)],
                        "answer": answer,
                        "tokens": tokens
                    })
                    print(f"✅ Request {request_id} completed: {answer[:30]}... ({tokens} tokens)")
                else:
                    # Failed request
                    results["failed"] += 1
                    if error and "Rate limit" in error:
                        results["rate_limited"] += 1
                    else:
                        results["other_errors"] += 1
            
            # Display batch metrics
            batch_time = time.time() - batch_start_time
            print(f"Batch completed in {batch_time:.2f} seconds.")
            
            # Get current queue status
            try:
                status = await client.get_queue_status()
                print(f"Current queue status: Requests: Minute {status['rate_limits']['requests']['minute']['display']}")
            except Exception as e:
                print(f"Could not get queue status: {e}")
            
            # Small delay between batches to make the output more readable
            if batch_end < total_requests:
                await asyncio.sleep(1)
        
        # Final queue status
        status = await client.get_queue_status()
        print("\nFinal queue status:")
        print(f"  Requests: Minute {status['rate_limits']['requests']['minute']['display']}, "
              f"Day {status['rate_limits']['requests']['day']['display']}")
        
        # Print summary
        elapsed = time.time() - start_time
        print("\n" + "="*80)
        print("TEST RESULTS")
        print("="*80)
        print(f"Total time: {elapsed:.2f} seconds")
        print(f"Total requests: {total_requests}")
        print(f"Completed: {results['completed']} ({results['completed']/total_requests*100:.1f}%)")
        print(f"Failed: {results['failed']} ({results['failed']/total_requests*100:.1f}%)")
        print(f"  Rate limited: {results['rate_limited']}")
        print(f"  Other errors: {results['other_errors']}")
        
        # Print some successful responses
        if results["responses"]:
            print("\nSample responses:")
            for i in range(min(3, len(results["responses"]))):
                resp = results["responses"][i]
                print(f"  Q: {resp['question']}")
                print(f"  A: {resp['answer'][:50]}..." if len(resp['answer']) > 50 else f"  A: {resp['answer']}")
                print()
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up
        await client.close()
        print("\nClient closed.")

if __name__ == "__main__":
    asyncio.run(test_rate_limits()) 