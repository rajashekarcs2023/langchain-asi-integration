import os
import sys
import json
from typing import Dict, List, Any, Optional, Tuple, Callable, Union, Annotated
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_asi import ChatASI

# Load environment variables
load_dotenv()

# Import project modules (adjust paths as needed)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.chain import create_rag_chain
from agents.search_agent_asi import create_search_agent
from agents.sec_agent_asi import create_sec_agent
from agents.supervisor_agent_asi import create_supervisor_agent

def create_research_graph():
    """Create a simplified research graph that ensures both agents are used.
    
    This implementation avoids the complex StateGraph structure and instead
    uses a simple sequential approach with explicit tracking of which agents
    have been visited.
    """
    # Make sure we have an API key
    asi_api_key = os.environ.get("ASI_API_KEY")
    if not asi_api_key:
        raise ValueError("ASI_API_KEY environment variable not set")
    
    # Initialize LLM with proper parameters
    llm = ChatASI(
        model_name="asi1-mini",  # Using ASI1 model
        temperature=0.7,
        max_tokens=2000,        # Increased token limit
        asi_api_key=asi_api_key, # Explicitly pass the API key
        streaming=False,         # Disable streaming for stability
        request_timeout=120.0     # Increase timeout to 120 seconds
    )
    
    # Create the RAG chain
    rag_chain = create_rag_chain()
    
    # Create agents
    supervisor_agent = create_supervisor_agent(llm)
    search_agent = create_search_agent(llm)
    sec_agent = create_sec_agent(llm, rag_chain)
    
    # Define the process function
    def process_query(query: str):
        """Process a financial query using both agents and combining their responses."""
        print(f"\n[RESEARCH GRAPH] Processing query: {query}")
        
        # Initialize state
        state = {
            "messages": [HumanMessage(content=query)],
            "team_members": ["Search", "SECAnalyst"],
            "visited": [],
            "conversation_flow": []
        }
        
        # Step 1: Ask supervisor which agent to call first
        print("\n[RESEARCH GRAPH] Step 1: Asking supervisor which agent to call first...")
        supervisor_result = supervisor_agent.invoke({
            "messages": state["messages"],
            "team_members": state["team_members"]
        })
        
        # Extract the next agent from the supervisor's response
        next_agent = supervisor_result.get("next", "SECAnalyst")
        print(f"\n[RESEARCH GRAPH] Supervisor recommends calling {next_agent} first")
        
        # Add supervisor to conversation flow
        state["conversation_flow"].append("Supervisor")
        
        # Step 2: Call the first agent
        print(f"\n[RESEARCH GRAPH] Step 2: Calling {next_agent}...")
        if next_agent == "SECAnalyst":
            agent_result = sec_agent.invoke({
                "messages": state["messages"],
                "team_members": state["team_members"]
            })
        else:  # Search
            agent_result = search_agent.invoke({
                "messages": state["messages"],
                "team_members": state["team_members"]
            })
        
        # Add the first agent to visited list and update conversation flow
        state["visited"].append(next_agent)
        state["conversation_flow"].append(next_agent)
        
        # Add the first agent's response to messages
        agent_content = agent_result.get('output', f"No analysis available from {next_agent}")
        agent_message = HumanMessage(
            content=f"{next_agent} has provided the following information:\n\n{agent_content}\n\nPlease proceed to the next step in the sequence.",
            name=next_agent
        )
        state["messages"].append(agent_message)
        
        # Step 3: Call the second agent
        second_agent = "SECAnalyst" if next_agent == "Search" else "Search"
        print(f"\n[RESEARCH GRAPH] Step 3: Calling {second_agent}...")
        if second_agent == "SECAnalyst":
            agent_result = sec_agent.invoke({
                "messages": state["messages"],
                "team_members": state["team_members"]
            })
        else:  # Search
            agent_result = search_agent.invoke({
                "messages": state["messages"],
                "team_members": state["team_members"]
            })
        
        # Add the second agent to visited list and update conversation flow
        state["visited"].append(second_agent)
        state["conversation_flow"].append(second_agent)
        
        # Add the second agent's response to messages
        agent_content = agent_result.get('output', f"No analysis available from {second_agent}")
        agent_message = HumanMessage(
            content=f"{second_agent} has provided the following information:\n\n{agent_content}\n\nPlease provide a final comprehensive analysis.",
            name=second_agent
        )
        state["messages"].append(agent_message)
        
        # Step 4: Call supervisor for final analysis
        print("\n[RESEARCH GRAPH] Step 4: Calling supervisor for final analysis...")
        final_result = supervisor_agent.invoke({
            "messages": state["messages"],
            "team_members": state["team_members"]
        })
        
        # Add supervisor to conversation flow
        state["conversation_flow"].append("Supervisor")
        
        # Extract the final analysis
        final_analysis = final_result.get("output", "No final analysis available")
        
        # Return the final result
        return {
            "output": final_analysis,
            "conversation_flow": state["conversation_flow"],
            "visited": state["visited"],
            "messages": state["messages"]
        }
    
    # Return the process function
    return process_query

# Example usage
def test_research_graph():
    """Test the research graph with a sample query."""
    print("\n===== Testing Research Graph =====\n")
    
    # Create the research graph
    research_graph = create_research_graph()
    
    # Define a test query with financial profile context
    test_query = (
        "What are the major supply chain risks for Apple, and how might they affect my investment? "
        "[Context: For Test User. Financial profile: risk tolerance: moderate; investment horizon: medium-term. "
        "Tailor the financial analysis to this person's profile and their interest in Apple investments.]"
    )
    
    print(f"Query: {test_query}")
    
    # Process the query
    result = research_graph(test_query)
    
    # Print the conversation flow
    print("\nConversation Flow:")
    if "conversation_flow" in result:
        print(" -> ".join(result["conversation_flow"]))
    else:
        print("No conversation flow found in the result.")
    
    # Print the visited agents
    print("\nVisited Agents:")
    if "visited" in result:
        print(", ".join(result["visited"]))
    else:
        print("No visited agents found in the result.")
    
    # Print the final analysis
    print("\n===== Final Analysis =====\n")
    print(result["output"])
    
    # Save the result to a file
    with open("research_graph_result.txt", "w") as f:
        f.write(f"Query: {test_query}\n\n")
        f.write(f"Conversation Flow: {' -> '.join(result['conversation_flow'])}\n\n")
        f.write(f"Visited Agents: {', '.join(result['visited'])}\n\n")
        f.write(f"Final Analysis:\n{result['output']}\n")
    
    print("\nResult saved to research_graph_result.txt")
    
    return result

# Run the test
if __name__ == "__main__":
    test_research_graph()
