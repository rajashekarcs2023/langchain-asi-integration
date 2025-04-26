import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_asi import ChatASI

# Load environment variables
load_dotenv()

# Get API key from environment
asi_api_key = os.environ.get("ASI_API_KEY")
print(f"API Key: {asi_api_key[:3]}...{asi_api_key[-5:]}")

# Initialize the chat model
llm = ChatASI(
    model_name="asi1-mini",
    temperature=0.7,
    max_tokens=1000,
    asi_api_key=asi_api_key,
    streaming=False,
    request_timeout=120.0
)

# Process a financial query using a sequential approach
def process_financial_query(query, persona=None):
    print(f"Processing query: {query}")
    
    # Add persona context if available
    persona_context = ""
    if persona:
        print(f"Using persona: {persona}")
        persona_context = f"Tailor your response for a client with {persona['data']['risk_tolerance']} risk tolerance and a {persona['data']['investment_horizon']} investment horizon."
    
    try:
        # Step 1: Get SEC data
        print("\n=== Step 1: Getting SEC data ===")
        sec_prompt = """You are an SEC Analyst specializing in analyzing regulatory filings.
        Focus on extracting key supply chain risks from Apple's latest 10-K filing.
        Be concise but comprehensive."""
        
        sec_messages = [
            SystemMessage(content=sec_prompt),
            HumanMessage(content=query)
        ]
        
        print("\n==== ASI API Request ====")
        print(f"URL: https://api.asi1.ai/v1/chat/completions")
        print(f"Headers: Authorization: Bearer {asi_api_key[:3]}...{asi_api_key[-5:]}")
        
        payload = {
            'model': 'asi1-mini',
            'stream': False,
            'n': 1,
            'temperature': 0.7,
            'max_tokens': 1000,
            'messages': [
                {
                    'content': sec_prompt,
                    'role': 'system'
                },
                {
                    'content': query,
                    'role': 'user'
                }
            ]
        }
        print(f"Payload: {json.dumps(payload, indent=2)}")
        print("========================")
        
        sec_response = llm.invoke(sec_messages)
        print(f"\nSEC Analysis:\n{sec_response.content}")
        
        # Step 2: Get market data
        print("\n=== Step 2: Getting market data ===")
        search_prompt = """You are a Market Analyst specializing in technology companies.
        Focus on recent analyst concerns about Apple's supply chain.
        Be concise but comprehensive."""
        
        search_messages = [
            SystemMessage(content=search_prompt),
            HumanMessage(content=query)
        ]
        
        print("\n==== ASI API Request ====")
        print(f"URL: https://api.asi1.ai/v1/chat/completions")
        print(f"Headers: Authorization: Bearer {asi_api_key[:3]}...{asi_api_key[-5:]}")
        
        payload = {
            'model': 'asi1-mini',
            'stream': False,
            'n': 1,
            'temperature': 0.7,
            'max_tokens': 1000,
            'messages': [
                {
                    'content': search_prompt,
                    'role': 'system'
                },
                {
                    'content': query,
                    'role': 'user'
                }
            ]
        }
        print(f"Payload: {json.dumps(payload, indent=2)}")
        print("========================")
        
        search_response = llm.invoke(search_messages)
        print(f"\nMarket Analysis:\n{search_response.content}")
        
        # Step 3: Combine the analyses
        print("\n=== Step 3: Final analysis ===")
        final_prompt = """You are a Financial Advisor helping a client with their investment decisions.
        Synthesize the SEC filing data and market analyst perspectives to provide a comprehensive analysis.
        Be concise but thorough.
        Tailor your advice to the client's risk tolerance and investment horizon."""
        
        final_messages = [
            SystemMessage(content=final_prompt),
            HumanMessage(content=f"SEC Filing Analysis:\n{sec_response.content}\n\nMarket Analyst Perspective:\n{search_response.content}\n\nBased on this information, how do Apple's supply chain risks in their latest 10-K filing compare to recent market analyst concerns? {persona_context}")
        ]
        
        final_response = llm.invoke(final_messages)
        print(f"\nFinal Analysis:\n{final_response.content}")
        
        # Save the complete response
        complete_response = {
            "query": query,
            "sec_analysis": sec_response.content,
            "market_analysis": search_response.content,
            "final_analysis": final_response.content
        }
        
        # Return the final response
        return complete_response
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# Main function to handle incoming messages
def on_message_callback(message):
    session_id = message.get("session_id", "unknown")
    query = message.get("content", "")
    
    print(f"Received: {query}")
    
    # Get persona for the session
    persona = {"name": "Test User", "data": {"risk_tolerance": "moderate", "investment_horizon": "medium-term"}}
    print(f"Persona for session {session_id}: {persona}")
    
    # Process the query
    result = process_financial_query(query, persona)
    
    # Format the response
    if "error" in result:
        response = f"I apologize, but I encountered an error while processing your query: {result['error']}"
    else:
        # Create a nicely formatted response
        response = f"## SEC Filing Analysis\n\n{result['sec_analysis']}\n\n## Market Analyst Perspective\n\n{result['market_analysis']}\n\n## Summary for Your Investment Profile\n\n{result['final_analysis']}"
    
    # Save the response
    os.makedirs("responses", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    response_file = f"responses/{session_id}_{timestamp}.json"
    
    with open(response_file, "w") as f:
        json.dump({
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "sec_analysis": result.get("sec_analysis", ""),
            "market_analysis": result.get("market_analysis", ""),
            "final_analysis": result.get("final_analysis", "")
        }, f, indent=2)
    
    print(f"Response saved to {response_file}")
    print(f"Sent: Financial analysis response\nsession_id: {session_id}")

# Test function
def test_main():
    print("\n===== Testing Standalone ASI Approach =====\n")
    
    # Create a test message
    test_message = {
        "session_id": "test-standalone",
        "id": 12345,
        "content": "How do Apple's supply chain risks in their latest 10-K filing compare to recent market analyst concerns?",
        "created_at": "2025-03-21T12:00:00"
    }
    
    # Process the message
    on_message_callback(test_message)
    
    print("\nTest completed successfully!")

# Run the test if this file is executed directly
if __name__ == "__main__":
    test_main()
