import os
import asyncio
from typing import Dict, List, Optional, Any

from langchain_asi.langchain_asi.chat_models import ChatASI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

# Initialize the chat model with streaming enabled
chat = ChatASI(
    model_name="asi1-mini",  # Will automatically use https://api.asi1.ai/v1
    streaming=True,
    verbose=True,  # Set to True to see API requests and responses
)

# Define a simple calculator tool
@tool
def calculator(expression: str) -> float:
    """Evaluate a mathematical expression."""
    try:
        return eval(expression)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

# Example of synchronous tool usage
def run_tool_example():
    print("\n===== Tool Usage Example =====\n")
    
    # Bind the calculator tool to the chat model
    chat_with_tools = chat.bind_tools([calculator])
    
    # Create a conversation with a math problem
    messages = [
        SystemMessage(content="You are a helpful AI assistant that's good at math."),
        HumanMessage(content="What is 123 * 456 divided by 7? Show your work.")
    ]
    
    # Generate a response
    response = chat_with_tools.invoke(messages)
    print(f"Response: {response.content}")
    
    # If there are tool calls, execute them and continue the conversation
    if response.tool_calls:
        print(f"\nTool calls detected: {response.tool_calls}")
        
        # Execute each tool call
        for tool_call in response.tool_calls:
            tool_name = tool_call.name
            tool_args = tool_call.args
            
            if tool_name == "calculator":
                result = calculator(tool_args)
                print(f"Calculator result: {result}")
                
                # Add the tool result to the conversation
                messages.append(response)
                messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call.id,
                    )
                )
                
                # Get the final response
                final_response = chat.invoke(messages)
                print(f"\nFinal response after tool call: {final_response.content}")

# Example of asynchronous streaming
async def run_async_streaming_example():
    print("\n===== Async Streaming Example =====\n")
    
    # Create a conversation
    messages = [
        SystemMessage(content="You are a helpful AI assistant."),
        HumanMessage(content="Write a short poem about artificial intelligence and creativity.")
    ]
    
    # Stream the response asynchronously
    print("Streaming response:")
    async for chunk in chat.astream(messages):
        print(chunk.content, end="", flush=True)
    print("\n")

# Example of handling multiple conversations concurrently
async def run_concurrent_conversations():
    print("\n===== Concurrent Conversations Example =====\n")
    
    # Create multiple conversations
    conversations = [
        [
            SystemMessage(content="You are a helpful AI assistant."),
            HumanMessage(content="Explain quantum computing in one paragraph.")
        ],
        [
            SystemMessage(content="You are a helpful AI assistant."),
            HumanMessage(content="What are the key features of Python 3.10?")
        ],
        [
            SystemMessage(content="You are a helpful AI assistant."),
            HumanMessage(content="Suggest a healthy breakfast recipe.")
        ]
    ]
    
    # Process all conversations concurrently
    async def process_conversation(idx, messages):
        print(f"Starting conversation {idx+1}...")
        response = await chat.ainvoke(messages)
        print(f"\nConversation {idx+1} response:\n{response.content}\n")
        return response
    
    # Gather all tasks
    tasks = [process_conversation(i, conv) for i, conv in enumerate(conversations)]
    responses = await asyncio.gather(*tasks)
    
    print(f"Completed {len(responses)} conversations concurrently.")

# Main function to run all examples
async def main():
    # Run the synchronous example
    run_tool_example()
    
    # Run the async examples
    await run_async_streaming_example()
    await run_concurrent_conversations()

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
