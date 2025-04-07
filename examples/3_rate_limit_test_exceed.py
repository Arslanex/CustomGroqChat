"""
Rate Limit Test (Exceed Limit)

This script sends multiple requests at once to intentionally exceed the rate limit
and demonstrate the rate limiting error handling.
"""
import asyncio
import time
from CustomGroqChat import GroqClient, RateLimitExceededException, CustomGroqChatException

# Very simple questions to minimize response time and token usage
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
    "What is 11+11?",
    "What is 12+12?",
    "What is 13+13?",
    "What is 14+14?",
    "What is 15+15?",
]

async def send_request_without_retry(client, model, message, request_id):
    """Send a request with no retry logic to show rate limit errors."""
    messages = [
        {"role": "system", "content": "You are a calculator. Just return the numeric answer with no explanation."},
        {"role": "user", "content": message}
    ]
    
    try:
        # Try to send the request
        response = await client.chat_completion(
            model_name=model,
            messages=messages,
            temperature=0.2
        )
        
        # Extract response
        answer = response["choices"][0]["message"]["content"]
        tokens = response.get("usage", {}).get("total_tokens", "N/A")
        
        print(f"✅ Request {request_id}: {message} → {answer} ({tokens} tokens)")
        return {"success": True, "request_id": request_id, "answer": answer, "tokens": tokens}
        
    except RateLimitExceededException as e:
        print(f"⚠️ Request {request_id}: Rate limit exceeded ({e.limit_type})")
        return {"success": False, "request_id": request_id, "error": str(e), "type": "rate_limit"}
        
    except CustomGroqChatException as e:
        print(f"❌ Request {request_id}: Error: {e}")
        return {"success": False, "request_id": request_id, "error": str(e), "type": "other"}

async def exceed_rate_limit():
    """Try to exceed the rate limit."""
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

        # Use the llama model which has 30 req/min limit
        model = "llama-3.1-8b-instant"
        if model not in models:
            print(f"Llama model not found, using: {models[0]}")
            model = models[0]
            
        print(f"Using model: {model}")
        
        # Attempt to exceed the rate limit
        num_requests = 35  # Exceeds the 30/min limit
        print(f"Making {num_requests} requests at once (rate limit should be 30/min)...")
        
        # Get initial status
        status = await client.get_queue_status()
        print(f"\nInitial queue status: {status['rate_limits']['requests']['minute']['display']} requests/min")
        
        # Create all tasks at once without waiting
        start_time = time.time()
        tasks = []
        
        for i in range(num_requests):
            question = QUESTIONS[i % len(QUESTIONS)]
            request_id = i + 1
            
            # Create task for this request
            task = asyncio.create_task(
                send_request_without_retry(client, model, question, request_id)
            )
            tasks.append(task)
            
            # Print what we're doing
            print(f"Queued request {request_id}: {question}")
        
        # Wait for all tasks to complete
        print("\nWaiting for all requests to complete...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze results
        elapsed = time.time() - start_time
        success_count = sum(1 for r in results if isinstance(r, dict) and r.get("success") == True)
        rate_limit_count = sum(1 for r in results if isinstance(r, dict) and r.get("type") == "rate_limit")
        other_error_count = sum(1 for r in results if isinstance(r, dict) and r.get("type") == "other")
        exception_count = sum(1 for r in results if isinstance(r, Exception))
        
        # Get final status
        try:
            status = await client.get_queue_status()
            final_status = status['rate_limits']['requests']['minute']['display']
        except:
            final_status = "Could not retrieve"
            
        # Print report
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        print(f"Completed in {elapsed:.2f} seconds")
        print(f"Total requests: {num_requests}")
        print(f"Successful: {success_count}")
        print(f"Rate limit errors: {rate_limit_count}")
        print(f"Other errors: {other_error_count}")
        print(f"Exceptions: {exception_count}")
        print(f"Final queue status: {final_status} requests/min")
        
    except Exception as e:
        print(f"Test error: {e}")
    finally:
        # Clean up
        await client.close()
        print("\nClient closed.")

if __name__ == "__main__":
    asyncio.run(exceed_rate_limit()) 