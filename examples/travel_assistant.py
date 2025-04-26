"""Travel assistant example using multiple tools with langchain-asi."""
import os
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any
from langchain_asi import ChatASI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
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
class SearchRestaurants(BaseModel):
    """Search for restaurants in a given location"""
    
    location: str = Field(description="The city and state, e.g. San Francisco, CA")
    cuisine: Optional[str] = Field(
        default=None,
        description="Type of cuisine, e.g. Italian, Mexican, etc."
    )
    price_range: Optional[str] = Field(
        default=None,
        description="Price range from $ to $$$$"
    )


class GetDirections(BaseModel):
    """Get directions from one location to another"""
    
    start: str = Field(description="Starting location, e.g. 123 Main St, San Francisco, CA")
    destination: str = Field(description="Destination location, e.g. Golden Gate Park, San Francisco, CA")
    mode: Optional[str] = Field(
        default="driving",
        description="Mode of transportation: driving, walking, bicycling, or transit"
    )


# Mock implementations of our tools
def search_restaurants(location: str, cuisine: Optional[str] = None, price_range: Optional[str] = None) -> str:
    """Mock implementation of restaurant search API"""
    cuisine_str = f" {cuisine}" if cuisine else ""
    price_str = f" ({price_range})" if price_range else ""
    print(f"[FOOD] Searching for{cuisine_str} restaurants in {location}{price_str}...")
    
    # In a real implementation, this would call a restaurant API
    restaurants = [
        {
            "name": f"{cuisine or 'Local'} Delight",
            "rating": 4.5,
            "price": price_range or "$$",
            "address": f"123 Main St, {location}",
            "cuisine": cuisine or "Local",
            "phone": "(555) 123-4567",
            "hours": "9:00 AM - 10:00 PM"
        },
        {
            "name": f"The {cuisine or 'Gourmet'} Experience",
            "rating": 4.7,
            "price": price_range or "$$$",
            "address": f"456 Oak Ave, {location}",
            "cuisine": cuisine or "Fusion",
            "phone": "(555) 987-6543",
            "hours": "11:00 AM - 11:00 PM"
        },
        {
            "name": f"{cuisine or 'City'} Bistro",
            "rating": 4.2,
            "price": price_range or "$$",
            "address": f"789 Pine St, {location}",
            "cuisine": cuisine or "International",
            "phone": "(555) 456-7890",
            "hours": "10:00 AM - 9:00 PM"
        }
    ]
    
    # Format the results as a string
    result = f"Found {len(restaurants)} restaurants in {location}:\n"
    for restaurant in restaurants:
        result += f"- {restaurant['name']}: {restaurant['rating']} stars, {restaurant['price']}, {restaurant['cuisine']} cuisine\n"
        result += f"  Address: {restaurant['address']}\n"
        result += f"  Phone: {restaurant['phone']}\n"
        result += f"  Hours: {restaurant['hours']}\n\n"
    
    return result


def get_directions(start: str, destination: str, mode: str = "driving") -> str:
    """Mock implementation of directions API"""
    print(f"[CAR] Getting {mode} directions from {start} to {destination}...")
    
    # In a real implementation, this would call a directions API
    directions = {
        "start": start,
        "destination": destination,
        "distance": "3.2 miles",
        "duration": "15 minutes",
        "mode": mode,
        "steps": [
            "Head north on Main St",
            "Turn right onto Oak Ave",
            "Continue for 2 miles",
            "Turn left onto Pine St",
            "Destination will be on your right"
        ]
    }
    
    # Format the results as a string
    result = f"Directions from {start} to {destination} ({mode}):\n"
    result += f"Total Distance: {directions['distance']}\n"
    result += f"Estimated Duration: {directions['duration']}\n\n"
    result += "Steps:\n"
    for i, step in enumerate(directions['steps'], 1):
        result += f"{i}. {step}\n"
    
    return result


def main():
    """Run the travel assistant example."""
    # Initialize the chat model
    chat = ChatASI(
        model_name="asi1-mini",
        temperature=0.1  # Lower temperature for more deterministic responses
    )
    
    # Bind multiple tools to the model
    chat_with_tools = chat.bind_tools([SearchRestaurants, GetDirections])
    
    # Create a conversation with system instructions
    messages = [
        SystemMessage(content="""You are a helpful travel assistant that can help users find restaurants and get directions.
        Always use the appropriate tool when a user asks for restaurant recommendations or directions.
        Do not make up information - always use the appropriate tool.
        Be detailed and helpful in your responses, providing useful information about the restaurants or directions."""),
        HumanMessage(content="I'm visiting Portland and would like to find some good Italian restaurants. After dinner, I'll need directions from downtown Portland to the Japanese Garden.")
    ]
    
    # Process the conversation
    print("\n[USER] I'm visiting Portland and would like to find some good Italian restaurants. After dinner, I'll need directions from downtown Portland to the Japanese Garden.")
    
    # Get the initial response
    response = chat_with_tools.invoke(messages)
    print("\n[ASSISTANT]:", response.content)
    
    # Check if the model wants to use tools
    if response.tool_calls:
        # Process the first tool call (restaurant search)
        tool_call = response.tool_calls[0]
        print(f"\n[TOOL CALL]: {tool_call['name']}")
        print(f"Arguments: {tool_call['args']}")
        
        # Execute the restaurant search tool
        if tool_call['name'] == "SearchRestaurants":
            tool_result = search_restaurants(
                location=tool_call['args'].get("location", "Portland, OR"),
                cuisine=tool_call['args'].get("cuisine"),
                price_range=tool_call['args'].get("price_range")
            )
        else:
            tool_result = f"Error: Unexpected first tool call {tool_call['name']}"
        
        # Add the tool result to the conversation
        messages.append(response)
        messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))
        
        # Get the next response
        response = chat_with_tools.invoke(messages)
        print("\n[ASSISTANT]:", response.content)
        
        # Check if the model wants to use another tool (directions)
        if response.tool_calls:
            # Process the second tool call (directions)
            tool_call = response.tool_calls[0]
            print(f"\n[TOOL CALL]: {tool_call['name']}")
            print(f"Arguments: {tool_call['args']}")
            
            # Execute the directions tool
            if tool_call['name'] == "GetDirections":
                tool_result = get_directions(
                    start=tool_call['args'].get("start", "Downtown Portland, OR"),
                    destination=tool_call['args'].get("destination", "Japanese Garden, Portland, OR"),
                    mode=tool_call['args'].get("mode", "driving")
                )
            else:
                tool_result = f"Error: Unexpected second tool call {tool_call['name']}"
            
            # Add the tool result to the conversation
            messages.append(response)
            messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))
            
            # Get the final response
            response = chat_with_tools.invoke(messages)
            print("\n[ASSISTANT]:", response.content)


if __name__ == "__main__":
    main()
