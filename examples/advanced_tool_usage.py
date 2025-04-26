"""Advanced tool usage example for langchain-asi."""
import os
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any
from langchain_asi import ChatASI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Check if ASI_API_KEY is set
if not os.environ.get("ASI_API_KEY"):
    print("Error: ASI_API_KEY environment variable not found.")
    print("Please create a .env file with your ASI_API_KEY or set it directly in your environment.")
    print("Example .env file content: ASI_API_KEY=your-api-key-here")
    exit(1)


# Define our tools using Pydantic models
class GetWeather(BaseModel):
    """Get the current weather in a given location"""
    
    location: str = Field(description="The city and state, e.g. San Francisco, CA")
    unit: Optional[str] = Field(
        default="fahrenheit", 
        description="The unit of temperature, either 'celsius' or 'fahrenheit'"
    )


class Calculator(BaseModel):
    """Perform a calculation"""
    
    operation: str = Field(description="The mathematical operation to perform: add, subtract, multiply, divide")
    a: float = Field(description="First number")
    b: float = Field(description="Second number")


# Mock implementations of our tools
def get_weather(location: str, unit: str = "fahrenheit") -> Dict[str, Any]:
    """Mock implementation of weather API"""
    print(f"🌤️ Getting weather for {location} in {unit}...")
    # In a real implementation, this would call a weather API
    weather_data = {
        "location": location,
        "temperature": 72 if unit == "fahrenheit" else 22,
        "condition": "Sunny",
        "humidity": "45%",
        "wind": "5 mph"
    }
    return weather_data


def calculate(operation: str, a: float, b: float) -> Dict[str, Any]:
    """Perform a calculation"""
    print(f"🧮 Calculating {a} {operation} {b}...")
    result = None
    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        if b == 0:
            return {"error": "Cannot divide by zero"}
        result = a / b
    else:
        return {"error": f"Unknown operation: {operation}"}
        
    return {
        "operation": operation,
        "a": a,
        "b": b,
        "result": result
    }


def simple_tool_example():
    """Demonstrate simple tool usage with ChatASI."""
    # Initialize the chat model
    chat = ChatASI(
        model_name="asi1-mini",
        temperature=0.1  # Lower temperature for more deterministic responses
    )
    
    # Bind the tool to the model
    chat_with_tools = chat.bind_tools([GetWeather])
    
    # Create a conversation with system instructions
    messages = [
        SystemMessage(content="""You are a helpful weather assistant that can check the weather for users.
        Always use the GetWeather tool when a user asks about the weather for a specific location.
        Do not make up weather information - always use the tool."""),
        HumanMessage(content="What's the weather like in Seattle today?")
    ]
    
    # Get the response from the model
    response = chat_with_tools.invoke(messages)
    print("\n🤖 Assistant:", response.content)
    
    # Check if there are any tool calls
    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(f"\n🔧 Tool Call: {tool_call['name']}")
            print(f"Arguments: {tool_call['args']}")
            
            # Execute the tool
            result = get_weather(
                location=tool_call['args']["location"],
                unit=tool_call['args'].get("unit", "fahrenheit")
            )
            
            print(f"\n🔧 Tool Result: {result}")
            
            # Add the tool response to the messages
            messages.append(AIMessage(content=str(result), tool_call_id=tool_call["id"]))
        
        # Continue the conversation with the tool results
        final_response = chat_with_tools.invoke(messages)
        print("\n🤖 Final Response:", final_response.content)
    else:
        print("\nNo tool calls were made.")


def multi_tool_example():
    """Demonstrate using multiple tools with ChatASI."""
    # Initialize the chat model
    chat = ChatASI(
        model_name="asi1-mini",
        temperature=0.1  # Lower temperature for more deterministic responses
    )
    
    # Bind multiple tools to the model
    chat_with_tools = chat.bind_tools([GetWeather, Calculator])
    
    # Create a conversation with system instructions
    messages = [
        SystemMessage(content="""You are a helpful assistant that can check the weather and perform calculations.
        Always use the appropriate tool when a user asks for weather information or to perform a calculation.
        Do not make up information - always use the appropriate tool.
        Handle one task at a time - first check the weather, then do the calculation."""),
        HumanMessage(content="I'm planning a trip to Miami. What's the weather like there? Also, I need to calculate how much I'll spend if I buy 3 tickets at $125 each.")
    ]
    
    # Process the conversation
    print("\n👤 User: I'm planning a trip to Miami. What's the weather like there? Also, I need to calculate how much I'll spend if I buy 3 tickets at $125 each.")
    
    # Get the initial response
    response = chat_with_tools.invoke(messages)
    print("\n🤖 Assistant:", response.content)
    
    # Process each tool call and continue the conversation
    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(f"\n🔧 Tool Call: {tool_call['name']}")
            print(f"Arguments: {tool_call['args']}")
            
            # Execute the appropriate tool
            if tool_call['name'] == "GetWeather":
                result = get_weather(
                    location=tool_call['args']["location"],
                    unit=tool_call['args'].get("unit", "fahrenheit")
                )
            elif tool_call['name'] == "Calculator":
                result = calculate(
                    operation=tool_call['args']["operation"],
                    a=float(tool_call['args']["a"]),
                    b=float(tool_call['args']["b"])
                )
            else:
                result = {"error": f"Unknown tool: {tool_call['name']}"}
            
            print(f"\n🔧 Tool Result: {result}")
            
            # Add the tool response to the messages
            messages.append(AIMessage(content=str(result), tool_call_id=tool_call["id"]))
        
        # Continue the conversation with the tool results
        final_response = chat_with_tools.invoke(messages)
        print("\n🤖 Final Response:", final_response.content)
        
        # Check if there are additional tool calls in the final response
        if final_response.tool_calls:
            print("\n🔧 Additional tool call detected. Processing...")
            for tool_call in final_response.tool_calls:
                print(f"\n🔧 Tool Call: {tool_call['name']}")
                print(f"Arguments: {tool_call['args']}")
                
                # Execute the appropriate tool
                if tool_call['name'] == "GetWeather":
                    result = get_weather(
                        location=tool_call['args']["location"],
                        unit=tool_call['args'].get("unit", "fahrenheit")
                    )
                elif tool_call['name'] == "Calculator":
                    result = calculate(
                        operation=tool_call['args']["operation"],
                        a=float(tool_call['args']["a"]),
                        b=float(tool_call['args']["b"])
                    )
                else:
                    result = {"error": f"Unknown tool: {tool_call['name']}"}
                
                print(f"\n🔧 Tool Result: {result}")
                
                # Add the tool response to the messages
                messages.append(AIMessage(content=str(result), tool_call_id=tool_call["id"]))
            
            # Get the final response after all tool calls
            complete_response = chat_with_tools.invoke(messages)
            print("\n🤖 Complete Response:", complete_response.content)
    else:
        print("\nNo tool calls were made.")


if __name__ == "__main__":
    print("\n🔧 Simple Tool Usage Example:\n")
    simple_tool_example()
    
    print("\n" + "-"*80 + "\n")
    
    print("🔧 Multi-Tool Example:\n")
    multi_tool_example()
