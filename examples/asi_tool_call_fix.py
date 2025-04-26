"""Demonstration of a fix for ASI function calling in LangChain."""

import os
import sys
import json
from dotenv import load_dotenv
from langchain_asi import ChatASI
from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

# Load environment variables
load_dotenv()

# Define a simple tool
@tool
def get_weather(location: str, unit: str = "celsius") -> str:
    """Get the weather for a specific location.
    
    Args:
        location: The city and state, e.g. San Francisco, CA
        unit: The temperature unit to use. 'celsius' or 'fahrenheit'
    
    Returns:
        The current weather for the specified location.
    """
    # This is a mock function that would normally call a weather API
    return f"The weather in {location} is sunny and 22 degrees {unit}."

# Create a patched version of ChatASI with improved tool call handling
class PatchedChatASI(ChatASI):
    """A patched version of ChatASI with improved tool call handling."""
    
    def _process_chat_response(self, response_data):
        """Process the chat response from the ASI API.
        
        This patched version better handles ASI's tool call format.
        """
        print("\nRaw API Response:")
        print(json.dumps(response_data, indent=2))
        
        # Extract choices from the response
        choices = response_data.get("choices", [])
        
        # Process each choice to create chat generations
        generations = []
        for choice in choices:
            message = choice.get("message", {})
            
            # Handle ASI-specific tool call format
            tool_calls = message.get("tool_calls", [])
            if tool_calls:
                print("\nFound tool calls in response:", tool_calls)
                
                # Convert ASI tool call format to OpenAI format
                converted_tool_calls = []
                for tool_call in tool_calls:
                    # Extract the tool call information
                    tool_call_id = tool_call.get("id", f"call_{len(converted_tool_calls)}")
                    tool_call_type = "function"  # Convert to expected type
                    
                    # Extract function information
                    function_info = {}
                    if "function" in tool_call:
                        function_info = tool_call["function"]
                    elif "name" in tool_call:
                        # Handle case where function info is directly in tool_call
                        function_info = {
                            "name": tool_call.get("name"),
                            "arguments": tool_call.get("arguments", "{}")
                        }
                    
                    # Create converted tool call
                    converted_tool_call = {
                        "id": tool_call_id,
                        "type": tool_call_type,
                        "function": function_info
                    }
                    converted_tool_calls.append(converted_tool_call)
                
                # Update message with converted tool calls
                message["tool_calls"] = converted_tool_calls
            
            # Create AIMessage from the processed message
            ai_message = AIMessage(
                content=message.get("content", ""),
                additional_kwargs={k: v for k, v in message.items() if k != "content"}
            )
            
            # Create ChatGeneration
            generation = ChatGeneration(
                message=ai_message,
                generation_info={
                    "finish_reason": choice.get("finish_reason"),
                    "logprobs": choice.get("logprobs")
                }
            )
            generations.append(generation)
        
        # Create and return ChatResult
        return ChatResult(generations=generations)

def test_patched_function_calling():
    """Test function calling with the patched ASI chat model."""
    print("Testing function calling with patched ASI...")
    
    # Initialize the patched chat model
    chat = PatchedChatASI(
        model_name="asi1-mini",  # Use the appropriate model
        temperature=0,
        verbose=True
    )
    
    # Bind the tool to the model
    chat_with_tools = chat.bind_tools(
        tools=[get_weather],
        tool_choice="auto"  # Let the model decide when to use the tool
    )
    
    # Create a message that should trigger tool use
    messages = [
        {"role": "system", "content": "You are a helpful assistant that can get weather information."},
        {"role": "user", "content": "What's the weather like in San Francisco?"}
    ]
    
    try:
        # Invoke the model with the messages
        response = chat_with_tools.invoke(messages)
        print("\nProcessed Response:")
        print(f"Type: {type(response)}")
        print(f"Content: {response.content}")
        
        # Check if tool calls were made
        tool_calls = response.additional_kwargs.get("tool_calls", [])
        print(f"\nTool calls: {len(tool_calls)}")
        for i, tool_call in enumerate(tool_calls):
            print(f"\nTool call {i+1}:")
            print(f"Type: {tool_call.get('type')}")
            print(f"Function: {tool_call.get('function', {}).get('name')}")
            print(f"Arguments: {tool_call.get('function', {}).get('arguments')}")
        
        return True
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_patched_function_calling()
    sys.exit(0 if success else 1)
