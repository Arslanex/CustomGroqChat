"""
Example 4: Conversation with Memory

This script demonstrates how to implement a multi-turn conversation
that maintains context memory, allowing the model to reference
previous parts of the conversation.
"""
import asyncio
import time
import os
import json
from datetime import datetime
from CustomGroqChat import GroqClient, CustomGroqChatException

class ConversationManager:
    """Manages a conversation with memory and history saving/loading."""
    
    def __init__(self, client, model_name, system_prompt=None):
        """Initialize the conversation manager."""
        self.client = client
        self.model_name = model_name
        self.system_prompt = system_prompt or "You are a helpful and friendly assistant."
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.save_dir = "conversations"
        self.latest_usage = {}
        
        # Create save directory if it doesn't exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    async def send_message(self, message_text):
        """Send a message to the model and get a response."""
        # Add user message to the conversation
        self.messages.append({"role": "user", "content": message_text})
        
        try:
            # Get response from the model
            response = await self.client.chat_completion(
                model_name=self.model_name,
                messages=self.messages,
                temperature=0.7
            )
            
            # Extract the assistant's message
            assistant_message = response["choices"][0]["message"]["content"]
            
            # Store usage stats
            if "usage" in response:
                self.latest_usage = response["usage"]
            
            # Add the assistant's response to the conversation
            self.messages.append({"role": "assistant", "content": assistant_message})
            
            return assistant_message
            
        except CustomGroqChatException as e:
            # If the error is related to token limits, try to reduce the context
            if "token limit" in str(e).lower():
                print("Warning: Token limit reached. Reducing conversation history...")
                # Keep system message and last 2 user messages for context
                if len(self.messages) > 5:
                    self.messages = [self.messages[0]] + self.messages[-4:]
                    # Try again with reduced context
                    return await self.send_message(message_text)
            
            # Re-raise other exceptions
            raise
    
    def save_conversation(self):
        """Save the conversation to a JSON file."""
        if len(self.messages) <= 1:
            print("No conversation to save.")
            return
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.save_dir, f"conversation_{timestamp}.json")
        
        # Prepare data to save
        data = {
            "model": self.model_name,
            "system_prompt": self.system_prompt,
            "messages": self.messages,
            "timestamp": timestamp
        }
        
        # Save to file
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"Conversation saved to {filename}")
        return filename
    
    @classmethod
    async def load_conversation(cls, client, filename):
        """Load a conversation from a file."""
        try:
            with open(filename, "r") as f:
                data = json.load(f)
            
            # Create a new conversation manager
            manager = cls(
                client=client,
                model_name=data.get("model"),
                system_prompt=data.get("system_prompt")
            )
            
            # Load messages
            manager.messages = data.get("messages", [])
            
            # If no system message, add one
            if not manager.messages or manager.messages[0]["role"] != "system":
                manager.messages.insert(0, {"role": "system", "content": manager.system_prompt})
            
            return manager
            
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error loading conversation: {e}")
            return None
    
    def display_history(self):
        """Display the conversation history."""
        for i, message in enumerate(self.messages):
            # Skip system message
            if i == 0 and message["role"] == "system":
                continue
                
            prefix = "You:" if message["role"] == "user" else "Assistant:"
            print(f"\n{prefix} {message['content']}")
    
    def get_token_usage(self):
        """Get the latest token usage statistics."""
        return self.latest_usage

async def main():
    """Run an interactive conversation with memory."""
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
        
        # Let user select a model
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
        
        # Create the conversation manager
        conversation = ConversationManager(client, selected_model)
        
        # Welcome message
        print("\n" + "="*80)
        print("CONVERSATION WITH MEMORY")
        print("="*80)
        print("Type your messages and chat with the AI.")
        print("Special commands:")
        print("  /help - Show this help message")
        print("  /save - Save the conversation")
        print("  /load - Load a conversation")
        print("  /history - Show conversation history")
        print("  /clear - Clear conversation history")
        print("  /exit - Exit the conversation")
        print("="*80)
        
        # Main conversation loop
        while True:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            # Check for commands
            if user_input.startswith("/"):
                command = user_input[1:].lower()
                
                if command == "exit":
                    break
                    
                elif command == "help":
                    print("\nSpecial commands:")
                    print("  /help - Show this help message")
                    print("  /save - Save the conversation")
                    print("  /load - Load a conversation")
                    print("  /history - Show conversation history")
                    print("  /clear - Clear conversation history")
                    print("  /exit - Exit the conversation")
                
                elif command == "save":
                    conversation.save_conversation()
                
                elif command == "load":
                    # List available conversations
                    files = [f for f in os.listdir(conversation.save_dir) 
                             if f.startswith("conversation_") and f.endswith(".json")]
                    
                    if not files:
                        print("No saved conversations found.")
                        continue
                    
                    print("\nAvailable conversations:")
                    for i, filename in enumerate(files, 1):
                        print(f"{i}. {filename}")
                    
                    try:
                        idx = int(input("\nSelect a conversation to load (or 0 to cancel): "))
                        if idx == 0:
                            continue
                        if 1 <= idx <= len(files):
                            path = os.path.join(conversation.save_dir, files[idx-1])
                            conversation = await ConversationManager.load_conversation(client, path)
                            if conversation:
                                print(f"Loaded conversation from {path}")
                                print("Conversation history:")
                                conversation.display_history()
                        else:
                            print("Invalid selection.")
                    except ValueError:
                        print("Invalid input.")
                
                elif command == "history":
                    conversation.display_history()
                
                elif command == "clear":
                    # Reset to just the system message
                    conversation.messages = [{"role": "system", "content": conversation.system_prompt}]
                    print("Conversation history cleared.")
                
                continue
            
            # Skip empty messages
            if not user_input:
                continue
            
            # Process the user's message
            try:
                start_time = time.time()
                
                # Show "thinking" indicator
                print("Assistant: ", end="", flush=True)
                
                # Get response from the model
                assistant_message = await conversation.send_message(user_input)
                
                # Calculate and display response time
                elapsed = time.time() - start_time
                
                # Print the response
                print(assistant_message)
                
                # Show token usage
                usage = conversation.get_token_usage()
                if usage:
                    print(f"\n[Response time: {elapsed:.2f}s | "
                          f"Tokens: {usage.get('prompt_tokens', 'N/A')} prompt + "
                          f"{usage.get('completion_tokens', 'N/A')} completion = "
                          f"{usage.get('total_tokens', 'N/A')} total]")
                
            except CustomGroqChatException as e:
                print(f"Error: {e}")
            except Exception as e:
                print(f"Unexpected error: {e}")
    
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up
        await client.close()
        print("\nClient closed.")

if __name__ == "__main__":
    asyncio.run(main()) 