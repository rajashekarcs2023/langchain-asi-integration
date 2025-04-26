import os
import asyncio
from dotenv import load_dotenv
from langchain_asi.chat_models import ChatASI
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

async def test_basic_chat():
    """Test basic chat functionality."""
    print("Testing basic chat...")
    chat = ChatASI(model_name="asi1-mini", temperature=0.7)
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is the capital of France?")
    ]
    response = await chat.agenerate(messages=[messages])
    print(f"Response: {response.generations[0].message.content}")
    print(f"Additional kwargs: {response.generations[0].message.additional_kwargs}")
    print(f"Usage metadata: {response.generations[0].message.usage_metadata}")
    print("Basic chat test completed.\n")

async def test_tool_calling():
    """Test tool calling functionality."""
    print("Testing tool calling...")
    chat = ChatASI(model_name="asi1-mini", temperature=0.7)
    
    # Define a weather tool
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What's the weather like in San Francisco?")
    ]
    
    response = await chat.agenerate(messages=[messages], tools=tools)
    print(f"Response: {response.generations[0].message.content}")
    print(f"Tool calls: {response.generations[0].message.tool_calls}")
    print(f"Additional kwargs: {response.generations[0].message.additional_kwargs}")
    print("Tool calling test completed.\n")

async def test_streaming():
    """Test streaming functionality."""
    print("Testing streaming...")
    chat = ChatASI(model_name="asi1-mini", temperature=0.7, streaming=True)
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Write a short poem about AI.")
    ]
    
    # Stream the response
    print("Streaming response:")
    async for chunk in chat.astream(messages):
        print(chunk.content, end="", flush=True)
    print("\nStreaming test completed.\n")

async def main():
    """Run all tests."""
    print("=" * 50)
    print("Testing ChatASI Implementation")
    print("=" * 50)
    
    await test_basic_chat()
    await test_tool_calling()
    await test_streaming()
    
    print("=" * 50)
    print("All tests completed successfully!")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(main())
