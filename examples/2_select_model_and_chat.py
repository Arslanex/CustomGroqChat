"""
Example 2: Select a Model and Chat

This script lets the user select a model from those available in the configuration
and then send a message to it.
"""
import asyncio
import sys
from CustomGroqChat import GroqClient, CustomGroqChatException

async def select_model_and_chat():
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
        
        # Get the user's message
        user_message = input("\nEnter your message to the model: ")
        
        # Create the messages array
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message}
        ]
        
        # Send the request
        print(f"\nSending request to {selected_model}...")
        response = await client.chat_completion(
            model_name=selected_model,
            messages=messages,
            temperature=0.7
        )
        
        # Extract and display the response
        content = response["choices"][0]["message"]["content"]
        
        print("\n" + "="*80)
        print(f"RESPONSE FROM {selected_model}:")
        print("="*80)
        print(content)
        print("="*80)
        
        # Display token usage if available
        if "usage" in response:
            usage = response["usage"]
            print(f"\nToken usage:")
            print(f"  Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
            print(f"  Completion tokens: {usage.get('completion_tokens', 'N/A')}")
            print(f"  Total tokens: {usage.get('total_tokens', 'N/A')}")
        
    except CustomGroqChatException as e:
        print(f"\nError: {e}")
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    finally:
        # Clean up
        await client.close()
        print("\nClient closed.")

if __name__ == "__main__":
    asyncio.run(select_model_and_chat()) 