"""Test script for the financial research graph using the OpenAICompatibleASI adapter."""
import os
import json
from typing import Dict, List, TypedDict, Annotated, Type
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_asi import ChatASI
from langgraph.graph import END, StateGraph

from langchain_asi.openai_adapter import OpenAICompatibleASI, LangGraphASIAdapter

# Load API key from environment variable
api_key = os.environ.get("ASI_API_KEY")
if not api_key:
    raise ValueError("ASI_API_KEY environment variable not set")

# Define the state schema
class ResearchState(TypedDict):
    """State for the financial research graph."""
    messages: List[HumanMessage | SystemMessage | AIMessage]
    context: Dict

# Define the route schema for the supervisor
class RouteSchema(BaseModel):
    """Schema for routing between agents."""
    next: str = Field(description="The next agent to call")
    reasoning: str = Field(description="Reasoning for the routing decision")
    information_needed: List[str] = Field(description="Information needed for the next step")

# Define the SEC Analyst agent
def sec_analyst(state: ResearchState) -> Dict:
    """SEC Analyst agent that provides information from SEC filings."""
    # Initialize the ASI model with our adapter
    chat = ChatASI(model_name="asi1-mini", asi_api_key=api_key)
    chat_adapter = OpenAICompatibleASI(chat)
    
    # Create the system message
    system_message = SystemMessage(
        content="""You are an SEC Analyst specializing in analyzing SEC filings.
        Provide key financial information from recent SEC filings for the company mentioned.
        Focus on revenue, profit margins, debt levels, and any risk factors mentioned.
        
        For this test, simulate that you have access to the 2024 10-K filing for Apple Inc.
        Provide realistic but simulated data based on what would typically be in such a filing."""
    )
    
    # Get the query from the state
    messages = state["messages"]
    
    # Call the model
    response = chat_adapter.invoke([system_message] + messages)
    
    # Format the response
    formatted_response = f"SEC ANALYST REPORT:\n\n{response.content}"
    
    # Return the updated state
    return {
        "messages": state["messages"] + [AIMessage(content=formatted_response, name="SECAnalyst")],
        "context": {**state["context"], "sec_data": formatted_response}
    }

# Define the Search agent
def search_agent(state: ResearchState) -> Dict:
    """Search agent that provides market context and news."""
    # Initialize the ASI model with our adapter
    chat = ChatASI(model_name="asi1-mini", asi_api_key=api_key)
    chat_adapter = OpenAICompatibleASI(chat)
    
    # Create the system message
    system_message = SystemMessage(
        content="""You are a Market Research specialist.
        Provide recent market context, news, and analyst opinions about the company mentioned.
        Focus on market trends, competitive landscape, and future outlook.
        
        For this test, simulate that you have access to recent market data for Apple Inc.
        Provide realistic but simulated data based on what would typically be available."""
    )
    
    # Get the query and SEC data from the state
    messages = state["messages"]
    sec_data = state["context"].get("sec_data", "")
    
    # Add the SEC data to the context
    context_message = HumanMessage(
        content=f"Here is the SEC filing data:\n\n{sec_data}\n\nPlease provide market context based on this information."
    )
    
    # Call the model
    response = chat_adapter.invoke([system_message] + messages + [context_message])
    
    # Format the response
    formatted_response = f"MARKET RESEARCH REPORT:\n\n{response.content}"
    
    # Return the updated state
    return {
        "messages": state["messages"] + [AIMessage(content=formatted_response, name="Search")],
        "context": {**state["context"], "market_data": formatted_response}
    }

# Define the final response generator
def generate_final_response(state: ResearchState) -> Dict:
    """Generate the final response based on all collected information."""
    # Initialize the ASI model with our adapter
    chat = ChatASI(model_name="asi1-mini", asi_api_key=api_key)
    chat_adapter = OpenAICompatibleASI(chat)
    
    # Create the system message
    system_message = SystemMessage(
        content="""You are a Financial Advisor providing recommendations to clients.
        Synthesize the SEC filing data and market research to provide a comprehensive analysis.
        Include a clear recommendation (Buy, Sell, or Hold) with supporting rationale.
        Consider both financial performance and market context in your analysis."""
    )
    
    # Get all the context from the state
    messages = state["messages"]
    sec_data = state["context"].get("sec_data", "")
    market_data = state["context"].get("market_data", "")
    
    # Create a summary message with all the data
    summary_message = HumanMessage(
        content=f"Based on the following information, provide a comprehensive financial analysis and recommendation:\n\n{sec_data}\n\n{market_data}"
    )
    
    # Call the model
    response = chat_adapter.invoke([system_message] + [summary_message])
    
    # Return the final response
    return {
        "messages": state["messages"] + [AIMessage(content=response.content)],
        "context": {**state["context"], "final_recommendation": response.content}
    }

# Build the graph
def build_graph():
    """Build the financial research graph."""
    # Initialize the ASI model for the supervisor
    chat = ChatASI(model_name="asi1-mini", asi_api_key=api_key, max_tokens=1000, streaming=False, request_timeout=120.0)
    
    # Create the LangGraph adapter
    adapter = LangGraphASIAdapter(chat)
    
    # Create the supervisor function
    supervisor = adapter.create_supervisor(
        system_prompt="""You are a supervisor coordinating a financial research process.
        Your job is to determine which specialist should be consulted next based on the current state of the research.
        
        The research process follows these steps:
        1. First, consult the SEC Analyst to get information from SEC filings
        2. Next, consult the Search specialist to get market context
        3. Finally, complete the process (FINISH)
        
        You must follow this exact sequence.""",
        members=["SECAnalyst", "Search"],
        schema=RouteSchema
    )
    
    # Create the graph
    graph = StateGraph(ResearchState)
    
    # Add the nodes
    graph.add_node("SECAnalyst", sec_analyst)
    graph.add_node("Search", search_agent)
    graph.add_node("FINISH", generate_final_response)
    
    # Add the conditional edges using the supervisor
    graph.add_conditional_edges(
        "SECAnalyst",
        lambda state: supervisor(state)["next"],  # Extract just the 'next' field from the result
        {"Search": "Search", "FINISH": "FINISH"}
    )
    
    graph.add_conditional_edges(
        "Search",
        lambda state: supervisor(state)["next"],  # Extract just the 'next' field from the result
        {"SECAnalyst": "SECAnalyst", "FINISH": "FINISH"}
    )
    
    # Add edge from FINISH to END
    graph.add_edge("FINISH", END)
    
    # Set the entry point
    graph.set_entry_point("SECAnalyst")
    
    # Compile the graph
    return graph.compile()

# Run the graph
def run_graph(query: str):
    """Run the financial research graph with the given query."""
    # Build the graph
    graph = build_graph()
    
    # Create the initial state
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "context": {"user_profile": "Conservative investor focused on long-term growth"}
    }
    
    # Run the graph
    result = graph.invoke(initial_state)
    
    return result

# Main function
def main():
    """Run the financial research graph with a sample query."""
    # Sample query
    query = "Analyze Apple's financial performance and provide a recommendation based on their 2024 10-K filing."
    
    print(f"\nProcessing query: {query}\n")
    print("Running financial research graph...\n")
    
    # Run the graph
    result = run_graph(query)
    
    # Print the messages
    print("\n===== CONVERSATION FLOW =====\n")
    for message in result["messages"]:
        if hasattr(message, "name") and message.name:
            print(f"\n[{message.name}]:\n{message.content}\n")
        else:
            if isinstance(message, HumanMessage):
                print(f"\n[USER]:\n{message.content}\n")
            else:
                print(f"\n[FINAL RECOMMENDATION]:\n{message.content}\n")
    
    # Print the final recommendation
    print("\n===== FINAL RECOMMENDATION =====\n")
    print(result["context"]["final_recommendation"])

# Run the main function
if __name__ == "__main__":
    main()
