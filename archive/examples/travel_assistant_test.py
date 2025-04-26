"""Test script for a travel assistant agent using the OpenAICompatibleASI adapter."""
import os
import json
from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_asi import ChatASI

from langchain_asi.openai_adapter import OpenAICompatibleASI

# Load API key from environment variable
api_key = os.environ.get("ASI_API_KEY")
if not api_key:
    raise ValueError("ASI_API_KEY environment variable not set")

# Define the travel recommendation schema
class TravelDestination(BaseModel):
    """Schema for travel destination recommendations."""
    destination: str = Field(description="Name of the recommended destination")
    country: str = Field(description="Country where the destination is located")
    best_time_to_visit: str = Field(description="The best time of year to visit this destination")
    budget_category: Literal["Budget", "Moderate", "Luxury"] = Field(description="Budget category for this destination")
    activities: List[str] = Field(description="List of recommended activities at this destination")
    safety_rating: int = Field(description="Safety rating from 1-10, with 10 being the safest")
    family_friendly: bool = Field(description="Whether the destination is suitable for families with children")
    visa_required: Optional[bool] = Field(description="Whether a visa is required for US citizens")

# Define the travel assistant function
def get_travel_recommendation(query: str) -> TravelDestination:
    """Get a travel recommendation based on the user's query."""
    # Initialize the ASI model with our adapter
    chat = ChatASI(model_name="asi1-mini", asi_api_key=api_key, max_tokens=1000, streaming=False, request_timeout=120.0)
    
    # Create the adapter with structured output
    chat_adapter = OpenAICompatibleASI(chat).with_structured_output(TravelDestination)
    
    # Create the system message with explicit JSON instructions
    system_message = SystemMessage(
        content="""You are a travel assistant that provides personalized travel recommendations.
        Based on the user's preferences, suggest a suitable destination with detailed information.
        
        IMPORTANT: You MUST respond with a valid JSON object containing these fields:
        - destination: Name of the recommended destination
        - country: Country where the destination is located
        - best_time_to_visit: The best time of year to visit
        - budget_category: Must be exactly one of: "Budget", "Moderate", or "Luxury"
        - activities: A list of strings with recommended activities
        - safety_rating: An integer from 1-10, with 10 being the safest
        - family_friendly: A boolean (true/false) indicating if it's good for families
        - visa_required: A boolean (true/false) or null if unknown
        
        Example response format:
        ```json
        {
          "destination": "Paris",
          "country": "France",
          "best_time_to_visit": "Spring (April to June) or Fall (September to October)",
          "budget_category": "Luxury",
          "activities": ["Visit the Eiffel Tower", "Explore the Louvre Museum", "Stroll along the Seine River"],
          "safety_rating": 8,
          "family_friendly": true,
          "visa_required": false
        }
        ```
        
        Do not include any explanatory text before or after the JSON object.
        """
    )
    
    # Create the user message
    user_message = HumanMessage(content=query)
    
    # Get the recommendation
    response = chat_adapter.invoke([system_message, user_message])
    
    # Check if we got a structured response
    if isinstance(response, TravelDestination):
        print(f"\nStructured Response Type: {type(response)}")
        print(f"Destination: {response.destination}")
        print(f"Country: {response.country}")
        print(f"Best Time to Visit: {response.best_time_to_visit}")
        print(f"Budget Category: {response.budget_category}")
        print(f"Activities: {', '.join(response.activities)}")
        print(f"Safety Rating: {response.safety_rating}/10")
        print(f"Family Friendly: {response.family_friendly}")
        print(f"Visa Required: {response.visa_required}")
        return response
    else:
        # If we got a regular message, try to extract JSON from it
        print(f"\nUnstructured Response Type: {type(response)}")
        if hasattr(response, 'content'):
            content = response.content
            # Try to find JSON in the content
            json_start = content.find('{')
            json_end = content.rfind('}')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                try:
                    json_str = content[json_start:json_end+1]
                    data = json.loads(json_str)
                    return TravelDestination(**data)
                except Exception as e:
                    print(f"Error extracting JSON: {e}")
        
        # If all else fails, return the response
        print(f"Could not extract structured data from response")
        return response

# Main function
def main():
    """Run the travel assistant with sample queries."""
    # Sample queries
    queries = [
        "I'm looking for a budget-friendly beach destination for a family vacation in the summer.",
        "I want to experience authentic culture and cuisine in Asia, preferably somewhere not too touristy.",
        "I'm planning a luxury honeymoon in Europe with beautiful scenery and romantic activities."
    ]
    
    # Test each query
    for i, query in enumerate(queries, 1):
        print(f"\n===== Test Query {i} =====\n")
        print(f"Query: {query}\n")
        
        try:
            recommendation = get_travel_recommendation(query)
            print(f"\nRecommendation successfully generated!")
        except Exception as e:
            print(f"\nError generating recommendation: {e}")

# Run the main function
if __name__ == "__main__":
    main()
