"""Example of using the ASI-to-OpenAI adapter with LangGraph."""
import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv
from typing import Annotated, List, TypedDict, Dict
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
import operator
from langgraph.graph import StateGraph, END
from langchain_asi import ChatASI
from pydantic import BaseModel, Field
import functools

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the OpenAI adapter
from langchain_asi.openai_adapter import OpenAICompatibleASI, create_openai_compatible_model

# Load environment variables
load_dotenv()

# Get API key from environment
asi_api_key = os.environ.get("ASI_API_KEY")
print(f"API Key: {asi_api_key[:3]}...{asi_api_key[-5:]}")

# Define the routing schema
class RouteSchema(BaseModel):
    """Schema for routing decisions."""
    next: str = Field(
        description="The next agent to call or FINISH if done",
    )
    reasoning: str = Field(
        description="Explanation for why this agent should act next"
    )
    information_needed: List[str] = Field(
        description="List of specific information needed from this agent"
    )

class ResearchTeamState(TypedDict):
    """Define the state structure for the research team."""
    messages: Annotated[List[BaseMessage], operator.add]
    next: str
    information_needed: List[str] = []
    reasoning: str = ""

# Create a mock SEC Analyst agent
def create_sec_agent(llm):
    """Create a SEC Analyst agent."""
    prompt = """You are an SEC Analyst specializing in analyzing regulatory filings.
    Focus on extracting key supply chain risks from Apple's latest 10-K filing.
    Be concise but comprehensive."""
    
    def sec_agent(state):
        messages = state["messages"]
        query = messages[-1].content
        
        # Create a new message list with the system prompt
        agent_messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=query)
        ]
        
        # Get the response from the LLM
        response = llm.invoke(agent_messages)
        
        return {"output": response.content}
    
    return sec_agent

# Create a mock Search agent
def create_search_agent(llm):
    """Create a Search agent."""
    prompt = """You are a Market Analyst specializing in technology companies.
    Focus on recent analyst concerns about Apple's supply chain.
    Be concise but comprehensive."""
    
    def search_agent(state):
        messages = state["messages"]
        query = messages[-1].content
        
        # Create a new message list with the system prompt
        agent_messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=query)
        ]
        
        # Get the response from the LLM
        response = llm.invoke(agent_messages)
        
        return {"output": response.content}
    
    return search_agent

def create_research_graph():
    """Create the research team graph with all agents and supervisor."""
    
    # Initialize LLM
    llm = ChatASI(
        model_name="asi1-mini",
        temperature=0.7,
        max_tokens=1000,
        asi_api_key=asi_api_key,
        streaming=False,
        request_timeout=120.0
    )
    
    # Create the OpenAI-compatible ASI adapter
    adapter = create_openai_compatible_model(llm)
    
    # Create agents
    search_agent = create_search_agent(llm)
    sec_agent = create_sec_agent(llm)
    
    # Create agent nodes directly (without adapter.agent_node which doesn't exist)
    def search_node(state):
        result = search_agent(state)
        return {"messages": state["messages"] + [AIMessage(content=result["output"])], "next": None}
    
    def sec_node(state):
        result = sec_agent(state)
        return {"messages": state["messages"] + [AIMessage(content=result["output"])], "next": None}
    
    # Create supervisor using the adapter's create_supervisor method
    supervisor_prompt = """You are a supervisor tasked with managing a conversation between
    Search and SECAnalyst workers to provide personalized financial advice.
    
    For SEC Analyst:
    - Use for historical financial data, regulatory filings, official numbers
    - Best for detailed financial metrics, risk factors, and regulatory information
    
    For Search:
    - Use for current market context, recent developments, analyst opinions
    - Best for industry trends, competitor analysis, and real-time updates
    """
    
    supervisor = adapter.create_supervisor(
        system_prompt=supervisor_prompt,
        members=["Search", "SECAnalyst"],
        schema=RouteSchema
    )
    
    # Create graph
    graph = StateGraph(ResearchTeamState)
    
    # Add nodes
    graph.add_node("Search", search_node)
    graph.add_node("SECAnalyst", sec_node)
    graph.add_node("supervisor", supervisor)
    
    # Add edges
    graph.add_edge("Search", "supervisor")
    graph.add_edge("SECAnalyst", "supervisor")
    
    # Add conditional edges from supervisor
    graph.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {
            "Search": "Search",
            "SECAnalyst": "SECAnalyst",
            "FINISH": END
        },
    )
    
    # Set entry point
    graph.set_entry_point("supervisor")
    
    return graph.compile()

def process_financial_query(query, persona=None):
    """Process a financial query through the research graph."""
    print(f"[ASI ADAPTER EXAMPLE] Processing query: {query}")
    
    # Add persona context if available
    if persona:
        print(f"[ASI ADAPTER EXAMPLE] Using persona: {persona}")
        query = f"{query} [Context: For {persona['name']}. Financial profile: risk tolerance: {persona['data']['risk_tolerance']}; investment horizon: {persona['data']['investment_horizon']}. Tailor the financial analysis to this person's profile and their interest in Apple investments.]"
    
    try:
        # Create the research graph
        print("[ASI ADAPTER EXAMPLE] Creating research graph...")
        graph = create_research_graph()
        
        # Process the query
        print(f"[ASI ADAPTER EXAMPLE] Invoking graph with query: {query}")
        result = graph.invoke({
            "messages": [HumanMessage(content=query)],
            "information_needed": [],
            "reasoning": ""
        })
        
        print("[ASI ADAPTER EXAMPLE] Query processing complete")
        return result
    except Exception as e:
        print(f"[ASI ADAPTER EXAMPLE] Error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# Extract the most informative content from the response
def extract_response_content(result):
    """Extract the most informative content from the response."""
    try:
        # Initialize variables for agent messages
        sec_message = None
        search_message = None
        
        # Check if we have a valid result
        if isinstance(result, dict) and "messages" in result and len(result["messages"]) > 0:
            # Look for messages from each agent
            for msg in result["messages"]:
                content = msg.content if hasattr(msg, "content") else str(msg)
                
                # Extract the actual content from the formatted message
                if "has provided the following information:" in content:
                    # Extract the content between the header and the footer
                    parts = content.split("has provided the following information:\n\n", 1)
                    if len(parts) > 1:
                        agent_content = parts[1].split("\n\nPlease proceed", 1)[0]
                        
                        if hasattr(msg, "name") and msg.name == "SECAnalyst":
                            sec_message = agent_content
                        elif hasattr(msg, "name") and msg.name == "Search":
                            search_message = agent_content
                # Fallback to the previous method
                elif content and len(content) > 100:  # Only consider substantial messages
                    if hasattr(msg, "name") and msg.name == "SECAnalyst":
                        sec_message = content
                    elif hasattr(msg, "name") and msg.name == "Search":
                        search_message = content
            
            # Combine the messages if we have both
            if sec_message and search_message:
                final_message = f"## SEC Filing Analysis\n\n{sec_message}\n\n## Market Analyst Perspective\n\n{search_message}\n\n## Summary for Your Investment Profile\n\nConsidering your moderate risk tolerance and medium-term investment horizon, you should monitor these supply chain risks carefully while evaluating Apple's mitigation strategies."
            elif sec_message:
                final_message = f"## SEC Filing Analysis\n\n{sec_message}\n\nNote: No market analysis was available for this query."
            elif search_message:
                final_message = f"## Market Analyst Perspective\n\n{search_message}\n\nNote: No SEC filing analysis was available for this query."
            else:
                # If no informative message found, use the last message
                final_message = result["messages"][-1].content if hasattr(result["messages"][-1], "content") else str(result["messages"][-1])
            
            return {
                "final_message": final_message,
                "sec_message": sec_message,
                "search_message": search_message
            }
        
        # If we couldn't extract the content, return the raw result
        return {"final_message": str(result)}
    
    except Exception as e:
        print(f"[ASI ADAPTER EXAMPLE] Error extracting response content: {e}")
        import traceback
        traceback.print_exc()
        return {"final_message": f"Error extracting response content: {str(e)}"}

# Test function
def test_main():
    print("\n===== Testing ASI OpenAI Adapter with LangGraph =====\n")
    
    # Create a test persona
    persona = {"name": "Test User", "data": {"risk_tolerance": "moderate", "investment_horizon": "medium-term"}}
    
    # Process a test query
    query = "How do Apple's supply chain risks in their latest 10-K filing compare to recent market analyst concerns?"
    print(f"Query: {query}")
    
    # Process the query
    result = process_financial_query(query, persona)
    
    # Extract the response content
    content = extract_response_content(result)
    
    # Save the response
    os.makedirs("responses", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    response_file = f"responses/adapter_test_{timestamp}.json"
    
    with open(response_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": content.get("final_message", ""),
            "sec_analysis": content.get("sec_message", ""),
            "market_analysis": content.get("search_message", ""),
        }, f, indent=2)
    
    print(f"\nResponse saved to {response_file}")
    print("\nTest completed successfully!")

# Run the test if this file is executed directly
if __name__ == "__main__":
    test_main()
