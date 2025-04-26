"""Trip planner example with multiple tool calls from a single query."""
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
class SearchHotels(BaseModel):
    """Search for hotels in a given location"""
    
    location: str = Field(description="The city and state, e.g. San Francisco, CA")
    check_in: str = Field(description="Check-in date in YYYY-MM-DD format")
    check_out: str = Field(description="Check-out date in YYYY-MM-DD format")
    guests: int = Field(description="Number of guests")
    max_price: Optional[int] = Field(
        default=None,
        description="Maximum price per night in USD"
    )


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


class SearchAttractions(BaseModel):
    """Search for tourist attractions in a given location"""
    
    location: str = Field(description="The city and state, e.g. San Francisco, CA")
    category: Optional[str] = Field(
        default=None,
        description="Category of attraction, e.g. museum, park, landmark"
    )
    family_friendly: Optional[bool] = Field(
        default=None,
        description="Whether the attraction is suitable for children"
    )


# Mock implementations of our tools
def search_hotels(location: str, check_in: str, check_out: str, guests: int, max_price: Optional[int] = None) -> str:
    """Mock implementation of hotel search API"""
    # In a real implementation, this would call a hotel API
    hotels = [
        {
            "name": "Grand Hotel",
            "rating": 4.5,
            "price": 250,
            "address": f"123 Main St, {location}",
            "amenities": ["Pool", "Spa", "Restaurant", "Free WiFi"],
            "available": True
        },
        {
            "name": "Boutique Inn",
            "rating": 4.7,
            "price": 180,
            "address": f"456 Oak Ave, {location}",
            "amenities": ["Free Breakfast", "Free WiFi", "Business Center"],
            "available": True
        },
        {
            "name": "Budget Lodge",
            "rating": 3.8,
            "price": 120,
            "address": f"789 Pine St, {location}",
            "amenities": ["Free WiFi", "Free Parking"],
            "available": True
        }
    ]
    
    # Filter by max price if provided
    if max_price:
        hotels = [hotel for hotel in hotels if hotel["price"] <= max_price]
    
    # Format the results as a string
    result = f"Found {len(hotels)} hotels in {location} for {guests} guests from {check_in} to {check_out}:\n"
    for hotel in hotels:
        result += f"- {hotel['name']}: {hotel['rating']} stars, ${hotel['price']}/night\n"
        result += f"  Address: {hotel['address']}\n"
        result += f"  Amenities: {', '.join(hotel['amenities'])}\n\n"
    
    return result


def search_restaurants(location: str, cuisine: Optional[str] = None, price_range: Optional[str] = None) -> str:
    """Mock implementation of restaurant search API"""
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


def search_attractions(location: str, category: Optional[str] = None, family_friendly: Optional[bool] = None) -> str:
    """Mock implementation of attractions search API"""
    # In a real implementation, this would call an attractions API
    attractions = [
        {
            "name": f"{category or 'City'} Museum",
            "rating": 4.6,
            "price": "$15",
            "address": f"123 Museum Rd, {location}",
            "category": category or "Museum",
            "family_friendly": True,
            "hours": "9:00 AM - 5:00 PM",
            "description": "A fascinating museum with exhibits for all ages."
        },
        {
            "name": f"{location} Park",
            "rating": 4.8,
            "price": "Free",
            "address": f"456 Park Ave, {location}",
            "category": "Park",
            "family_friendly": True,
            "hours": "6:00 AM - 10:00 PM",
            "description": "Beautiful urban park with walking trails and playgrounds."
        },
        {
            "name": f"Historic {location} Theater",
            "rating": 4.4,
            "price": "$25",
            "address": f"789 Broadway, {location}",
            "category": "Entertainment",
            "family_friendly": False,
            "hours": "Varies by performance",
            "description": "Historic theater featuring plays, musicals, and concerts."
        }
    ]
    
    # Filter by category if provided
    if category:
        attractions = [attr for attr in attractions if attr["category"].lower() == category.lower()]
    
    # Filter by family-friendly if provided
    if family_friendly is not None:
        attractions = [attr for attr in attractions if attr["family_friendly"] == family_friendly]
    
    # Format the results as a string
    result = f"Found {len(attractions)} attractions in {location}:\n"
    for attraction in attractions:
        result += f"- {attraction['name']}: {attraction['rating']} stars, {attraction['price']}\n"
        result += f"  Address: {attraction['address']}\n"
        result += f"  Category: {attraction['category']}\n"
        result += f"  Family-friendly: {'Yes' if attraction['family_friendly'] else 'No'}\n"
        result += f"  Hours: {attraction['hours']}\n"
        result += f"  Description: {attraction['description']}\n\n"
    
    return result


def process_user_query(query: str) -> str:
    """Process a user query and return the final response.
    
    This function handles all the tool calls behind the scenes and only returns
    the final response to the user.
    """
    # Initialize the chat model
    chat = ChatASI(
        model_name="asi1-mini",
        temperature=0.1  # Lower temperature for more deterministic responses
    )
    
    # Bind multiple tools to the model
    chat_with_tools = chat.bind_tools([SearchHotels, SearchRestaurants, SearchAttractions])
    
    # Create a conversation with system instructions
    messages = [
        SystemMessage(content="""You are a helpful trip planner that can help users plan their vacations.
        You can search for hotels, restaurants, and attractions in a given location.
        Always use the appropriate tool when a user asks for information about hotels, restaurants, or attractions.
        Do not make up information - always use the appropriate tool.
        Be detailed and helpful in your responses, providing useful information about the user's trip.
        Create a comprehensive trip plan that includes accommodations, dining options, and activities."""),
        HumanMessage(content=query)
    ]
    
    # Get the initial response
    response = chat_with_tools.invoke(messages)
    
    # If no tool calls, return the response directly
    if not response.tool_calls:
        return response.content
    
    # Process all tool calls until we get a final response
    while response.tool_calls:
        # Process each tool call
        for tool_call in response.tool_calls:
            # Execute the appropriate tool
            if tool_call['name'] == "SearchHotels":
                # Convert max_price to int if it exists
                max_price = tool_call['args'].get("max_price")
                if max_price and isinstance(max_price, str):
                    try:
                        max_price = int(max_price)
                    except ValueError:
                        max_price = None
                
                # Convert guests to int if it exists
                guests = tool_call['args'].get("guests", 2)
                if isinstance(guests, str):
                    try:
                        guests = int(guests)
                    except ValueError:
                        guests = 2
                
                tool_result = search_hotels(
                    location=tool_call['args'].get("location", "San Francisco, CA"),
                    check_in=tool_call['args'].get("check_in", "2023-07-01"),
                    check_out=tool_call['args'].get("check_out", "2023-07-05"),
                    guests=guests,
                    max_price=max_price
                )
            elif tool_call['name'] == "SearchRestaurants":
                # Get location from arguments or use the same location as previous tools
                location = tool_call['args'].get("location")
                if not location:
                    # Try to find location from previous tool calls
                    for prev_msg in messages:
                        if hasattr(prev_msg, 'tool_calls') and prev_msg.tool_calls:
                            for prev_tool in prev_msg.tool_calls:
                                prev_location = prev_tool['args'].get('location')
                                if prev_location:
                                    location = prev_location
                                    break
                            if location:
                                break
                
                # Default to San Diego if no location found
                if not location:
                    location = "San Diego, CA"
                    
                # Default cuisine to Mexican based on the user query
                cuisine = tool_call['args'].get("cuisine")
                if not cuisine and "Mexican" in query:
                    cuisine = "Mexican"
                
                tool_result = search_restaurants(
                    location=location,
                    cuisine=cuisine,
                    price_range=tool_call['args'].get("price_range")
                )
            elif tool_call['name'] == "SearchAttractions":
                # Get location from arguments or use the same location as previous tools
                location = tool_call['args'].get("location")
                if not location:
                    # Try to find location from previous tool calls
                    for prev_msg in messages:
                        if hasattr(prev_msg, 'tool_calls') and prev_msg.tool_calls:
                            for prev_tool in prev_msg.tool_calls:
                                prev_location = prev_tool['args'].get('location')
                                if prev_location:
                                    location = prev_location
                                    break
                            if location:
                                break
                
                # Default to San Diego if no location found
                if not location:
                    location = "San Diego, CA"
                
                # Convert family_friendly to bool if it exists
                family_friendly = tool_call['args'].get("family_friendly")
                if family_friendly is not None:
                    if isinstance(family_friendly, str):
                        family_friendly = family_friendly.lower() == "true"
                else:
                    # Default to True since user asked for family-friendly attractions
                    family_friendly = True
                
                tool_result = search_attractions(
                    location=location,
                    category=tool_call['args'].get("category"),
                    family_friendly=family_friendly
                )
            else:
                tool_result = f"Error: Unknown tool {tool_call['name']}"
            
            # Add the tool result to the conversation
            messages.append(response)
            messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))
        
        # Get the next response
        response = chat_with_tools.invoke(messages)
    
    # Return the final response
    return response.content


def main():
    """Run the trip planner example with only the final output shown to the user."""
    # User query that requires multiple tool calls
    user_query = """I'm planning a family vacation to San Diego from July 15-20, 2023. 
    We need a hotel for 2 adults and 2 kids under $300 per night. 
    We'd like to try some good Mexican restaurants while we're there. 
    Also, what are some family-friendly attractions we can visit?"""
    
    print("\n=== Trip Planner Demo ===\n")
    print("User:", user_query)
    print("\nProcessing...")
    
    # Process the query and get the final response
    final_response = process_user_query(user_query)
    
    # Show only the final response to the user
    print("\nAssistant:", final_response)
    
    print("\n=== Behind the Scenes ===\n")
    print("This query required the assistant to:")
    print("1. Search for hotels in San Diego within budget")
    print("2. Find Mexican restaurants in San Diego")
    print("3. Discover family-friendly attractions in San Diego")
    print("4. Combine all this information into a comprehensive trip plan")


if __name__ == "__main__":
    main()
