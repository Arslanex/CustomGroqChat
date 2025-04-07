"""
Example 1: Compare All Models

This script loads the configuration, runs the same prompt through all available models,
and displays the responses side by side for comparison.
"""
import asyncio
import json
import time
from CustomGroqChat import GroqClient, CustomGroqChatException

async def run_all_models():
    # Initialize the client
    print("Initializing client...")
    client = GroqClient()
    await client.initialize()
    
    try:
        # Get all available models
        models = client.get_available_models()
        if not models:
            print("No models found in configuration!")
            return
            
        print(f"Found {len(models)} models: {', '.join(models)}")
        
        # The prompt to send to all models
        prompt = "Explain the concept of quantum computing in one paragraph."
        messages = [
            {"role": "system", "content": "You are a helpful assistant that provides accurate information."},
            {"role": "user", "content": prompt}
        ]
        
        # Store results for each model
        results = {}
        
        # Process each model in sequence
        for model in models:
            print(f"\nSending request to {model}...")
            start_time = time.time()
            
            try:
                # Send the same prompt to each model
                response = await client.chat_completion(
                    model_name=model,
                    messages=messages,
                    temperature=0.7
                )
                
                # Extract response
                content = response["choices"][0]["message"]["content"]
                elapsed = time.time() - start_time
                
                # Store the result
                results[model] = {
                    "content": content,
                    "elapsed_time": elapsed,
                    "tokens": response.get("usage", {}).get("total_tokens", "N/A")
                }
                
                print(f"✓ {model} responded in {elapsed:.2f} seconds")
                
            except CustomGroqChatException as e:
                print(f"✗ Error with {model}: {e}")
                results[model] = {"error": str(e)}
        
        # Display the results
        print("\n" + "="*80)
        print("RESULTS COMPARISON")
        print("="*80)
        
        for model, data in results.items():
            print(f"\n## {model} ##")
            if "error" in data:
                print(f"ERROR: {data['error']}")
            else:
                print(f"Response time: {data['elapsed_time']:.2f} seconds")
                print(f"Total tokens: {data['tokens']}")
                print("-"*40)
                print(data["content"])
            print("-"*80)
        
        # Optional: Save results to a file
        with open("model_comparison_results.json", "w") as f:
            json.dump(results, f, indent=2)
            print("\nResults saved to model_comparison_results.json")
            
    finally:
        # Clean up
        await client.close()
        print("\nClient closed.")

if __name__ == "__main__":
    asyncio.run(run_all_models()) 