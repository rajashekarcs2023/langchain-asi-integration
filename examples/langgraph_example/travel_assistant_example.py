"""Travel assistant example using ChatASI with LangGraph."""
import os
from typing import Any, Dict, List, Optional, TypedDict, Annotated, Literal, Union
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_asi import ChatASI
from langgraph.graph import StateGraph, END
import operator
import json
from datetime import datetime, timedelta

# Set your API key - for a real implementation, use environment variables
asi_api_key = os.environ.get("ASI_API_KEY")
if not asi_api_key:
    raise ValueError("ASI_API_KEY environment variable is not set")

# Define tool schemas
class SearchHotels(BaseModel):
    """Search for hotels in a given location"""
    location: str = Field(description="The city and state, e.g. San Francisco, CA")
    check_in_date: Optional[str] = Field(default=None, description="Check-in date in YYYY-MM-DD format")
    check_out_date: Optional[str] = Field(default=None, description="Check-out date in YYYY-MM-DD format")
    guests: Optional[int] = Field(default=2, description="Number of guests")
    max_price: Optional[int] = Field(default=None, description="Maximum price per night in USD")
    family_friendly: Optional[bool] = Field(default=None, description="Whether the hotel should be family-friendly")

class SearchRestaurants(BaseModel):
    """Search for restaurants in a given location"""
    location: str = Field(description="The city and state, e.g. San Francisco, CA")
    cuisine: Optional[str] = Field(default=None, description="Type of cuisine")
    price_range: Optional[str] = Field(default=None, description="Price range, e.g. $, $$, $$$")
    family_friendly: Optional[bool] = Field(default=None, description="Whether the restaurant should be family-friendly")

class SearchAttractions(BaseModel):
    """Search for attractions in a given location"""
    location: str = Field(description="The city and state, e.g. San Francisco, CA")
    type: Optional[str] = Field(default=None, description="Type of attraction, e.g. museum, park, etc.")
    family_friendly: Optional[bool] = Field(default=None, description="Whether the attraction should be family-friendly")

# Define the state type
class AgentState(TypedDict):
    """State for the agent graph."""
    messages: List[BaseMessage]
    next: str
    # Track the outputs from each search
    hotels_output: Optional[str]
    restaurants_output: Optional[str]
    attractions_output: Optional[str]
    # Track the conversation history
    conversation_history: Annotated[List[Dict[str, str]], operator.add]

# Mock data for tools
def mock_search_hotels(location: str, **kwargs) -> str:
    """Mock function to search for hotels."""
    if "san diego" in location.lower():
        return """Here are some family-friendly hotels in San Diego:

1. **Hotel del Coronado** - This iconic beachfront resort offers family-friendly amenities including pools, beach activities, and spacious rooms. It's located on Coronado Island with stunning ocean views. Price range: $$$-$$$$

2. **Manchester Grand Hyatt San Diego** - Located downtown with easy access to attractions, this hotel features multiple pools, a kids' club, and family suites. Price range: $$$

3. **Catamaran Resort Hotel and Spa** - Situated on Mission Bay, this Polynesian-themed resort offers a private beach, water activities, and is close to SeaWorld. Great for families! Price range: $$$

4. **Hilton San Diego Resort & Spa** - Located on Mission Bay with a private beach, pools, and water sports rentals. Family-friendly with spacious rooms and suites. Price range: $$$

5. **Legoland California Resort Hotel** - If you're planning to visit Legoland, staying at their themed hotel would be perfect for children. It's located in Carlsbad, about 30 minutes from downtown San Diego. Price range: $$$"""
    else:
        return f"No hotels found in {location}."

def mock_search_restaurants(location: str, **kwargs) -> str:
    """Mock function to search for restaurants."""
    if "san diego" in location.lower():
        return """Here are some family-friendly restaurants in San Diego:

1. **The Crack Shack** - A casual, open-air eatery specializing in fried chicken. Kids love the playground area while parents enjoy craft beers. Located in Little Italy.

2. **Corvette Diner** - A 50's themed restaurant with singing waitstaff, arcade games, and a magic show. The fun atmosphere and comfort food make it perfect for families. Located in Liberty Station.

3. **Lucha Libre Taco Shop** - A colorful, wrestling-themed Mexican restaurant with delicious tacos and burritos. Kids love the champion's booth and luchador masks. Located in Mission Hills.

4. **Waypoint Public** - Features a dedicated kids' play area with books and toys, allowing parents to enjoy craft beers and upscale pub food. Located in North Park.

5. **Pizza Port** - A local chain with locations in Ocean Beach and Carlsbad, offering great pizza and craft beers in a casual setting perfect for families."""
    else:
        return f"No restaurants found in {location}."

def mock_search_attractions(location: str, **kwargs) -> str:
    """Mock function to search for attractions."""
    if "san diego" in location.lower():
        return """Here are some family-friendly attractions in San Diego:

1. **San Diego Zoo** - World-famous zoo in Balboa Park with over 12,000 animals and 650 species. The kids will love the guided bus tour and Skyfari aerial tram.

2. **SeaWorld San Diego** - Marine theme park with exciting rides, animal shows, and educational exhibits featuring dolphins, orcas, and other sea creatures.

3. **LEGOLAND California** - Located in nearby Carlsbad, this theme park is perfect for children aged 2-12 with LEGO-themed rides, shows, and attractions.

4. **Balboa Park** - A 1,200-acre cultural park with 17 museums, gardens, and the San Diego Zoo. The Model Railroad Museum and Fleet Science Center are particularly popular with kids.

5. **USS Midway Museum** - Explore this historic aircraft carrier with self-guided audio tours, flight simulators, and restored aircraft. Great for families with older children."""
    else:
        return f"No attractions found in {location}."

# Initialize the chat model
# The ASI API key should be set in your environment variables as ASI_API_KEY
chat = ChatASI(model_name="asi1-mini")

# Define the supervisor node
def supervisor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Supervisor node that coordinates the travel planning."""
    # Create the prompt template
    prompt = ChatPromptTemplate.from_template(
        """You are a travel planning assistant. Your job is to:
1. Understand the user's travel request
2. Determine what information to search for next
3. Provide reasoning for your decision
4. When all necessary information is gathered, create a comprehensive travel plan

Available search options:
- SearchHotels: Find accommodations in the destination
- SearchRestaurants: Find dining options in the destination
- SearchAttractions: Find activities and attractions in the destination

Question: {question}

Current conversation history:
{conversation_history}

Hotels information: {hotels_output}
Restaurants information: {restaurants_output}
Attractions information: {attractions_output}

Your response must be in the following format:

Thinking: Your step-by-step reasoning process
Next: [SearchHotels/SearchRestaurants/SearchAttractions/FINISH]
Reasoning: Brief explanation of why you chose this next step"""
    )
    
    # Get the messages from the state
    messages = state["messages"]
    question = messages[0].content if messages else ""
    hotels_output = state.get("hotels_output", "Not available yet")
    restaurants_output = state.get("restaurants_output", "Not available yet")
    attractions_output = state.get("attractions_output", "Not available yet")
    
    # Format the conversation history
    conversation_history = ""
    for entry in state.get("conversation_history", []):
        conversation_history += f"\n{entry['role']}: {entry['content']}\n"
    
    # Create the chain
    chain = prompt | chat
    
    # Invoke the chain
    response = chain.invoke({
        "question": question,
        "conversation_history": conversation_history,
        "hotels_output": hotels_output,
        "restaurants_output": restaurants_output,
        "attractions_output": attractions_output
    })
    
    # Parse the response
    content = response.content
    
    # Extract the next step
    next_step = "FINISH"  # Default
    for line in content.split("\n"):
        if line.startswith("Next:"):
            next_step = line.replace("Next:", "").strip()
    
    # Add to conversation history
    conversation_entry = {"role": "Supervisor", "content": content}
    
    # If we have all outputs and this is at least the second iteration, default to FINISH
    if (hotels_output != "Not available yet" and 
        restaurants_output != "Not available yet" and 
        attractions_output != "Not available yet" and 
        len(state.get("conversation_history", [])) >= 3):
        next_step = "FINISH"
    
    # Update the state
    return {
        "messages": messages + [response],
        "next": next_step,
        "conversation_history": [conversation_entry]
    }

# Define the search hotels node
def search_hotels_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node that searches for hotels."""
    # Get the original question
    question = state["messages"][0].content
    
    # Extract location from the question
    location = "San Diego, CA"  # Default for mock example
    if "in" in question:
        parts = question.split("in")
        if len(parts) > 1:
            location_part = parts[1].strip().split(".")[0].split("?")[0].split("!")[0]
            location = location_part
    
    # Set default parameters
    today = datetime.now()
    check_in = (today + timedelta(days=30)).strftime("%Y-%m-%d")
    check_out = (today + timedelta(days=37)).strftime("%Y-%m-%d")
    
    # Perform the mock search
    search_results = mock_search_hotels(
        location=location,
        check_in_date=check_in,
        check_out_date=check_out,
        guests=4,  # Assuming a family of 4
        family_friendly=True
    )
    
    # Create a prompt for the AI to summarize the results
    prompt = ChatPromptTemplate.from_template(
        """You are a hotel expert. Please summarize the following hotel search results for a family vacation:

{search_results}

Provide a helpful summary focusing on family-friendly features, locations, and price ranges."""
    )
    
    # Create the chain
    chain = prompt | chat
    
    # Invoke the chain
    response = chain.invoke({"search_results": search_results})
    
    # Add to conversation history
    conversation_entry = {"role": "HotelSearch", "content": response.content}
    
    # Update the state
    return {
        "messages": state["messages"] + [response],
        "hotels_output": response.content,
        "conversation_history": [conversation_entry]
    }

# Define the search restaurants node
def search_restaurants_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node that searches for restaurants."""
    # Get the original question
    question = state["messages"][0].content
    
    # Extract location from the question
    location = "San Diego, CA"  # Default for mock example
    if "in" in question:
        parts = question.split("in")
        if len(parts) > 1:
            location_part = parts[1].strip().split(".")[0].split("?")[0].split("!")[0]
            location = location_part
    
    # Perform the mock search
    search_results = mock_search_restaurants(
        location=location,
        family_friendly=True
    )
    
    # Create a prompt for the AI to summarize the results
    prompt = ChatPromptTemplate.from_template(
        """You are a restaurant expert. Please summarize the following restaurant search results for a family vacation:

{search_results}

Provide a helpful summary focusing on family-friendly features, cuisines, and locations."""
    )
    
    # Create the chain
    chain = prompt | chat
    
    # Invoke the chain
    response = chain.invoke({"search_results": search_results})
    
    # Add to conversation history
    conversation_entry = {"role": "RestaurantSearch", "content": response.content}
    
    # Update the state
    return {
        "messages": state["messages"] + [response],
        "restaurants_output": response.content,
        "conversation_history": [conversation_entry]
    }

# Define the search attractions node
def search_attractions_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node that searches for attractions."""
    # Get the original question
    question = state["messages"][0].content
    
    # Extract location from the question
    location = "San Diego, CA"  # Default for mock example
    if "in" in question:
        parts = question.split("in")
        if len(parts) > 1:
            location_part = parts[1].strip().split(".")[0].split("?")[0].split("!")[0]
            location = location_part
    
    # Perform the mock search
    search_results = mock_search_attractions(
        location=location,
        family_friendly=True
    )
    
    # Create a prompt for the AI to summarize the results
    prompt = ChatPromptTemplate.from_template(
        """You are an attractions expert. Please summarize the following attractions search results for a family vacation:

{search_results}

Provide a helpful summary focusing on family-friendly features, age appropriateness, and locations."""
    )
    
    # Create the chain
    chain = prompt | chat
    
    # Invoke the chain
    response = chain.invoke({"search_results": search_results})
    
    # Add to conversation history
    conversation_entry = {"role": "AttractionSearch", "content": response.content}
    
    # Update the state
    return {
        "messages": state["messages"] + [response],
        "attractions_output": response.content,
        "conversation_history": [conversation_entry]
    }

# Define the final response node
def final_response_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node that creates the final travel plan."""
    # Get the original question and all search outputs
    question = state["messages"][0].content
    hotels_output = state.get("hotels_output", "No hotel information available.")
    restaurants_output = state.get("restaurants_output", "No restaurant information available.")
    attractions_output = state.get("attractions_output", "No attraction information available.")
    
    # Create a prompt for the AI to create a comprehensive travel plan
    prompt = ChatPromptTemplate.from_template(
        """You are a travel planning expert. Please create a comprehensive travel plan based on the following information:

Original Request: {question}

Hotels Information:
{hotels_output}

Restaurants Information:
{restaurants_output}

Attractions Information:
{attractions_output}

Create a well-organized travel plan that includes recommended accommodations, dining options, and activities. Include a suggested itinerary for a 5-day trip."""
    )
    
    # Create the chain
    chain = prompt | chat
    
    # Invoke the chain
    response = chain.invoke({
        "question": question,
        "hotels_output": hotels_output,
        "restaurants_output": restaurants_output,
        "attractions_output": attractions_output
    })
    
    # Add to conversation history
    conversation_entry = {"role": "FinalPlan", "content": response.content}
    
    # Update the state
    return {
        "messages": state["messages"] + [response],
        "conversation_history": [conversation_entry]
    }

# Build the graph
def build_graph():
    """Build the agent graph."""
    # Create the graph
    graph = StateGraph(AgentState)
    
    # Add the nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("search_hotels", search_hotels_node)
    graph.add_node("search_restaurants", search_restaurants_node)
    graph.add_node("search_attractions", search_attractions_node)
    graph.add_node("final_response", final_response_node)
    
    # Add the conditional edges from supervisor
    graph.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {
            "SearchHotels": "search_hotels",
            "SearchRestaurants": "search_restaurants",
            "SearchAttractions": "search_attractions",
            "FINISH": "final_response"
        }
    )
    
    # Add the edges back to the supervisor
    graph.add_edge("search_hotels", "supervisor")
    graph.add_edge("search_restaurants", "supervisor")
    graph.add_edge("search_attractions", "supervisor")
    
    # Add the edge from final_response to END
    graph.add_edge("final_response", END)
    
    # Set the entry point
    graph.set_entry_point("supervisor")
    
    # Compile the graph
    return graph.compile()

# Run the graph
def run_graph(question: str):
    """Run the agent graph."""
    # Build the graph
    graph = build_graph()
    
    # Initialize the state
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "next": "",
        "hotels_output": None,
        "restaurants_output": None,
        "attractions_output": None,
        "conversation_history": [{"role": "Human", "content": question}]
    }
    
    # Run the graph with a recursion limit
    try:
        result = graph.invoke(initial_state, {"recursion_limit": 10})
        return result
    except Exception as e:
        print(f"Error: {e}")
        # Return the partial state if available
        return initial_state

# Format the conversation
def format_conversation(state: Dict[str, Any]) -> str:
    """Format the conversation for display."""
    conversation = ""
    for entry in state.get("conversation_history", []):
        conversation += f"\n{entry['role']}: {entry['content']}\n"
    return conversation

# Example usage
if __name__ == "__main__":
    # Example question
    question = "I'm planning a family vacation to San Diego. Can you help me find hotels, restaurants, and attractions?"
    
    # Run the graph
    result = run_graph(question)
    
    # Print the conversation history
    print("\nConversation History:")
    print(format_conversation(result))
    
    # Print the final answer
    if len(result["messages"]) > 1:
        print("\nFinal Travel Plan:")
        print(result["messages"][-1].content)
