"""
Custom Advanced Features

This script demonstrates how to use advanced features like tools and structured output
with the ChatASI implementation using custom implementations that don't rely on
LangChain's callback system.
"""

import os
import json
import asyncio
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List, Type, Union
from pydantic import BaseModel, Field
from langchain_asi.chat_models import ChatASI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

# Load environment variables
load_dotenv()

# Define a simple Pydantic model for structured output
class WeatherInfo(BaseModel):
    """Information about the weather in a location."""
    
    location: str = Field(description="The location for the weather information")
    temperature: float = Field(description="The temperature in Celsius")
    conditions: str = Field(description="The weather conditions (e.g., sunny, rainy)")
    humidity: Optional[float] = Field(None, description="The humidity percentage")

# Define a custom ChatASI class with methods for advanced features
class CustomChatASI(ChatASI):
    """Custom ChatASI implementation with methods for advanced features."""
    
    async def use_tool(self, messages: List[Dict[str, Any]], tool: Union[Dict[str, Any], Type[BaseModel]]):
        """Use a tool with the ChatASI model.
        
        Args:
            messages: A list of message dictionaries to send to the ASI API.
            tool: A tool definition to use with the ChatASI model.
                Can be a dictionary or a Pydantic model.
                
        Returns:
            The response from the ASI API.
        """
        # Convert the tool to the format expected by the ASI API
        formatted_tool = convert_to_openai_tool(tool)
        asi_tool = {
            "type": "function",
            "function": formatted_tool["function"]
        }
        
        # Make a direct API call with the tool
        response = await self.acompletion_with_retry(
            messages=messages,
            tools=[asi_tool],
            tool_choice="auto"
        )
        
        return response
    
    async def use_structured_output(self, messages: List[Dict[str, Any]], schema: Type[BaseModel]):
        """Use structured output with the ChatASI model.
        
        Args:
            messages: A list of message dictionaries to send to the ASI API.
            schema: A Pydantic model to use for structured output.
                
        Returns:
            The response from the ASI API.
        """
        # Convert the schema to the format expected by the ASI API
        formatted_tool = convert_to_openai_tool(schema)
        asi_tool = {
            "type": "function",
            "function": formatted_tool["function"]
        }
        
        # Make a direct API call with the tool
        response = await self.acompletion_with_retry(
            messages=messages,
            tools=[asi_tool],
            tool_choice={"type": "function", "function": {"name": schema.__name__}}
        )
        
        return response
    
    async def use_json_mode(self, messages: List[Dict[str, Any]]):
        """Use JSON mode with the ChatASI model.
        
        Args:
            messages: A list of message dictionaries to send to the ASI API.
                
        Returns:
            The response from the ASI API.
        """
        # Make a direct API call with JSON mode
        response = await self.acompletion_with_retry(
            messages=messages,
            response_format={"type": "json_object"}
        )
        
        return response

async def test_custom_advanced_features():
    """Test the custom advanced features implementation."""
    print("=" * 80)
    print("TESTING CUSTOM ADVANCED FEATURES")
    print("=" * 80)
    
    # Initialize the custom ChatASI model
    asi_chat = CustomChatASI(model_name="asi1-mini", verbose=True)
    
    # Test 1: use_tool
    print("\n[Test 1: use_tool]")
    try:
        # Create messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant that can provide weather information."},
            {"role": "user", "content": "What's the weather like in San Francisco?"}
        ]
        
        # Define a simple tool
        class GetWeather(BaseModel):
            """Get the current weather in a given location."""
            
            location: str = Field(description="The city and state, e.g. San Francisco, CA")
        
        # Use the tool
        response = await asi_chat.use_tool(messages, GetWeather)
        
        print(f"Response: {json.dumps(response, indent=2)}")
        
        # Extract tool calls from the response
        if "choices" in response and len(response["choices"]) > 0:
            choice = response["choices"][0]
            assistant_message = choice.get("message", {})
            
            if "tool_calls" in assistant_message:
                # Add the assistant message with tool_calls to our conversation
                messages.append({
                    "role": "assistant",
                    "content": assistant_message.get("content", ""),
                    "tool_calls": assistant_message.get("tool_calls", [])
                })
                
                tool_calls = assistant_message.get("tool_calls", [])
                print("\nTool calls:")
                
                for tool_call in tool_calls:
                    print(f"  Tool: {tool_call.get('function', {}).get('name')}")
                    print(f"  Arguments: {tool_call.get('function', {}).get('arguments')}")
                    
                    # Parse the arguments
                    args = json.loads(tool_call.get('function', {}).get('arguments', "{}"))
                    print(f"  Parsed arguments: {args}")
                    
                    # Simulate getting weather data
                    weather_data = {
                        "location": args.get("location", "Unknown"),
                        "temperature": 22.5,
                        "conditions": "Sunny",
                        "humidity": 65.0
                    }
                    
                    # Create a response to the tool call
                    tool_response = {
                        "role": "tool",
                        "tool_call_id": tool_call.get("id"),
                        "content": json.dumps(weather_data)
                    }
                    
                    # Add the tool response to the messages
                    messages.append(tool_response)
                
                # Get the final response
                final_response = await asi_chat.acompletion_with_retry(
                    messages=messages
                )
                
                print("\nFinal response:")
                if "choices" in final_response and len(final_response["choices"]) > 0:
                    final_content = final_response["choices"][0].get("message", {}).get("content", "")
                    print(f"  {final_content}")
            else:
                print("\nNo tool calls were made.")
                print(f"Message content: {assistant_message.get('content')}")
    except Exception as e:
        print(f"Error in use_tool test: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: use_structured_output
    print("\n[Test 2: use_structured_output]")
    try:
        # Create messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant that can provide weather information. Use the WeatherInfo tool to structure your response about weather conditions."},
            {"role": "user", "content": "Provide weather information for Paris with temperature, conditions, and humidity."}
        ]
        
        # Use structured output
        response = await asi_chat.use_structured_output(messages, WeatherInfo)
        
        print(f"Response: {json.dumps(response, indent=2)}")
        
        # Extract structured output from the response
        if "choices" in response and len(response["choices"]) > 0:
            choice = response["choices"][0]
            assistant_message = choice.get("message", {})
            
            if "tool_calls" in assistant_message:
                # Add the assistant message with tool_calls to our conversation
                messages.append({
                    "role": "assistant",
                    "content": assistant_message.get("content", ""),
                    "tool_calls": assistant_message.get("tool_calls", [])
                })
                
                tool_calls = assistant_message.get("tool_calls", [])
                print("\nStructured output:")
                
                for tool_call in tool_calls:
                    if tool_call.get("function", {}).get("name") == "WeatherInfo":
                        args_str = tool_call.get("function", {}).get("arguments", "{}")
                        args = json.loads(args_str)
                        
                        # Create a WeatherInfo object from the arguments
                        try:
                            weather_info = WeatherInfo(**args)
                            print(f"  {weather_info}")
                            
                            # Simulate a tool response
                            tool_response = {
                                "role": "tool",
                                "tool_call_id": tool_call.get("id"),
                                "content": json.dumps({"result": "Weather information processed successfully"})
                            }
                            
                            # Add the tool response to the messages
                            messages.append(tool_response)
                            
                            # Get the final response
                            final_response = await asi_chat.acompletion_with_retry(
                                messages=messages
                            )
                            
                            print("\nFinal response after structured output:")
                            if "choices" in final_response and len(final_response["choices"]) > 0:
                                final_content = final_response["choices"][0].get("message", {}).get("content", "")
                                print(f"  {final_content}")
                        except Exception as e:
                            print(f"  Error creating WeatherInfo object: {e}")
                            print(f"  Arguments: {args}")
            else:
                print("\nNo structured output was generated.")
                print(f"Message content: {assistant_message.get('content')}")
    except Exception as e:
        print(f"Error in use_structured_output test: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: use_json_mode
    print("\n[Test 3: use_json_mode]")
    try:
        # Create messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant that provides responses in JSON format."},
            {"role": "user", "content": "Provide a weather forecast for Paris, London, and New York."}
        ]
        
        # Use JSON mode
        response = await asi_chat.use_json_mode(messages)
        
        print(f"Response: {json.dumps(response, indent=2)}")
        
        # Extract JSON content from the response
        if "choices" in response and len(response["choices"]) > 0:
            choice = response["choices"][0]
            content = choice.get("message", {}).get("content", "")
            
            print("\nJSON content:")
            try:
                # Handle markdown code blocks (ASI-specific format)
                if content.startswith("```json") and content.endswith("```"):
                    # Extract the JSON content from the markdown code block
                    json_str = content.replace("```json", "").replace("```", "").strip()
                    json_content = json.loads(json_str)
                    print(f"  Successfully parsed JSON from markdown code block")
                    print(f"  {json.dumps(json_content, indent=2)}")
                else:
                    # Try to parse as regular JSON
                    json_content = json.loads(content)
                    print(f"  {json.dumps(json_content, indent=2)}")
            except json.JSONDecodeError as e:
                print(f"  Could not parse JSON: {e}")
                print(f"  Raw content: {content}")
    except Exception as e:
        print(f"Error in use_json_mode test: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    asyncio.run(test_custom_advanced_features())
