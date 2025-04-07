"""
Example 5: Parallel Processing with Callbacks

This script demonstrates how to efficiently process multiple requests in parallel
using callbacks, allowing for high-throughput applications when working with
large volumes of data.
"""
import asyncio
import time
import csv
import os
from typing import Dict, Any, List
from CustomGroqChat import GroqClient, CustomGroqChatException

# Sample data to process
SAMPLE_DATA = [
    {"id": 1, "text": "What is machine learning?"},
    {"id": 2, "text": "Explain the concept of blockchain."},
    {"id": 3, "text": "What are the benefits of cloud computing?"},
    {"id": 4, "text": "How does natural language processing work?"},
    {"id": 5, "text": "What is the Internet of Things?"},
    {"id": 6, "text": "Explain the difference between AI and machine learning."},
    {"id": 7, "text": "What are microservices in software architecture?"},
    {"id": 8, "text": "How do neural networks learn?"},
    {"id": 9, "text": "What is edge computing?"},
    {"id": 10, "text": "Explain containerization in DevOps."}
]

class ParallelProcessor:
    """Process multiple requests in parallel using callbacks."""
    
    def __init__(self, client, model_name, max_concurrent=3):
        """Initialize the processor with client and model."""
        self.client = client
        self.model_name = model_name
        self.max_concurrent = max_concurrent  # Maximum concurrent requests
        self.results = {}  # Store results by ID
        self.errors = {}  # Store errors by ID
        self.callback_futures = {}  # Track futures for callbacks
        self.semaphore = asyncio.Semaphore(max_concurrent)  # Control concurrency
        self.in_progress = 0  # Track number of requests in progress
        self.completed = 0  # Track number of completed requests
        self.start_time = None  # Track overall processing time

    async def process_item(self, item_id: int, text: str) -> None:
        """Process a single item and store the result."""
        async with self.semaphore:  # Limit concurrent requests
            self.in_progress += 1
            print(f"🔄 Processing item {item_id}: {text[:30]}... ({self.in_progress} active requests)")
            
            # Create messages for this item
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Provide concise answers to questions."},
                {"role": "user", "content": text}
            ]
            
            # Create a future to receive the callback result
            callback_future = asyncio.Future()
            self.callback_futures[item_id] = callback_future
            
            try:
                # Define the callback function
                async def response_callback(response: Dict[str, Any]) -> None:
                    if "error" in response:
                        callback_future.set_exception(Exception(response["error"]))
                    else:
                        callback_future.set_result(response)
                
                # Request ID for this request
                request_id = await self.client.request_handler.prepare_chat_request(
                    model_name=self.model_name,
                    messages=messages,
                    temperature=0.5,  # Use lower temperature for more consistent results
                    callback=response_callback,
                    priority="normal"  # Can also use "high" for important requests
                )
                
                print(f"📋 Queued request {request_id} for item {item_id}")
                
                # Wait for the callback to be triggered
                response = await callback_future
                
                # Extract the content
                content = response["choices"][0]["message"]["content"]
                tokens = response.get("usage", {}).get("total_tokens", "N/A")
                
                # Store the result
                self.results[item_id] = {
                    "content": content,
                    "tokens": tokens,
                    "request_id": request_id
                }
                
                self.completed += 1
                print(f"✅ Completed item {item_id} ({self.completed}/{len(SAMPLE_DATA)})")
                
            except Exception as e:
                # Store any errors
                self.errors[item_id] = str(e)
                print(f"❌ Error processing item {item_id}: {e}")
            finally:
                self.in_progress -= 1
                # Clean up
                if item_id in self.callback_futures:
                    del self.callback_futures[item_id]

    async def process_all(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process all items in parallel with controlled concurrency."""
        self.start_time = time.time()
        
        # Create tasks for all items
        tasks = []
        for item in items:
            task = asyncio.create_task(self.process_item(item["id"], item["text"]))
            tasks.append(task)
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks)
        
        elapsed = time.time() - self.start_time
        
        # Return results summary
        return {
            "elapsed_time": elapsed,
            "total_items": len(items),
            "completed": len(self.results),
            "errors": len(self.errors),
            "results": self.results,
            "error_details": self.errors
        }
    
    def export_to_csv(self, filename: str) -> None:
        """Export results to a CSV file."""
        if not self.results:
            print("No results to export.")
            return
            
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['id', 'question', 'answer', 'tokens']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for item in SAMPLE_DATA:
                item_id = item["id"]
                row = {
                    'id': item_id,
                    'question': item["text"],
                    'answer': self.results.get(item_id, {}).get("content", f"ERROR: {self.errors.get(item_id, 'Unknown error')}"),
                    'tokens': self.results.get(item_id, {}).get("tokens", "N/A")
                }
                writer.writerow(row)
                
        print(f"Results exported to {filename}")

async def main():
    """Run the parallel processing example."""
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
            
        # Select a model
        selected_model = models[0]  # Use first model for simplicity
        print(f"Using model: {selected_model}")
        
        # Create processor with selected model
        processor = ParallelProcessor(
            client=client, 
            model_name=selected_model,
            max_concurrent=3  # Process 3 requests at a time
        )
        
        # Display queue status before processing
        status = await client.get_queue_status()
        print("\nInitial queue status:")
        print(f"Queue lengths: {status['queue_lengths']}")
        print(f"Rate limits: {status['rate_limits']['requests']['minute']['display']} requests/min, "
              f"{status['rate_limits']['tokens']['minute']['display']} tokens/min")
        
        # Start processing
        print(f"\nProcessing {len(SAMPLE_DATA)} items with max {processor.max_concurrent} concurrent requests...")
        
        # Process all items
        results = await processor.process_all(SAMPLE_DATA)
        
        # Display results
        print("\n" + "="*80)
        print("PROCESSING RESULTS")
        print("="*80)
        print(f"Processed {results['total_items']} items in {results['elapsed_time']:.2f} seconds")
        print(f"Completed: {results['completed']}")
        print(f"Errors: {results['errors']}")
        print(f"Average time per item: {results['elapsed_time'] / results['total_items']:.2f} seconds")
        
        # Show some sample results
        print("\nSample results:")
        for i in range(1, min(4, len(processor.results) + 1)):
            if i in processor.results:
                print(f"\nItem {i}: {SAMPLE_DATA[i-1]['text']}")
                print(f"Response: {processor.results[i]['content'][:100]}...")
        
        # Export results to CSV
        processor.export_to_csv("parallel_processing_results.csv")
        
        # Display final queue status
        status = await client.get_queue_status()
        print("\nFinal queue status:")
        print(f"Queue lengths: {status['queue_lengths']}")
        print(f"Rate limits: {status['rate_limits']['requests']['minute']['display']} requests/min, "
              f"{status['rate_limits']['tokens']['minute']['display']} tokens/min")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up
        await client.close()
        print("\nClient closed.")

if __name__ == "__main__":
    asyncio.run(main()) 