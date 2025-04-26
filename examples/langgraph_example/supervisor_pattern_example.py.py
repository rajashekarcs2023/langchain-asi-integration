"""Example of using agents with LangGraph."""
import os
from typing import Any, Dict, List, Optional, TypedDict, Literal
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_asi import ChatASI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

# Set your API key - for a real implementation, use environment variables
os.environ["ASI_API_KEY"] = "sk_491aa37d22cc490883508f47e0c76e7abda7b212ab0642989937690bbd73a0b3"  # Replace with your actual API key

# Define the state
class AgentState(TypedDict):
    """State for the agent graph."""
    messages: List[Any]
    next: str
    reasoning: str
    information_needed: List[str]

# Mock data for search and SEC analyst tools
MOCK_SEARCH_DATA = {
    "tesla": """Tesla, Inc. is an American multinational automotive and clean energy company headquartered in Austin, Texas. Tesla designs and manufactures electric vehicles (cars and trucks), stationary battery energy storage devices, solar panels and solar roof tiles, and related products and services. As of 2023, Tesla is one of the world's most valuable companies and is one of the world's largest companies by market capitalization. In 2022, the company had the most worldwide sales of battery electric vehicles, with a market share of 18%.

Recent financial performance:
- Q4 2023 revenue: $25.17 billion
- 2023 annual revenue: $96.77 billion
- Vehicle deliveries in 2023: 1.81 million
- Current market cap: ~$600 billion
- Major products: Model S, Model 3, Model X, Model Y, Cybertruck, Powerwall, Solar Roof

Recent developments:
- Cybertruck production began in late 2023
- Expansion of Gigafactories in Texas, Berlin, and Shanghai
- Continued development of Full Self-Driving technology
- Energy business growing with Powerwall and commercial installations""",
    
    "financial prospects": """Tesla's financial prospects for the next year are influenced by several factors:

1. Electric Vehicle Market Growth: The global EV market is expected to grow by approximately 35% in the next year, with Tesla positioned to maintain its market leadership despite increasing competition.

2. Production Capacity: Tesla's expanded production capacity at its Gigafactories is expected to support delivery growth of 20-30% in the next year.

3. Margins: Automotive gross margins have faced pressure due to price adjustments, but are expected to stabilize as production efficiencies improve.

4. Energy Business: Tesla's energy generation and storage business is growing rapidly, with expectations of 40-50% growth in the next year.

5. Regulatory Environment: Government incentives for EVs continue in many markets, though some regions are scaling back subsidies.

6. Competition: Traditional automakers and new EV startups are launching competitive models, potentially pressuring Tesla's market share.

7. New Products: The ramp-up of Cybertruck production and potential new model announcements could drive additional revenue growth."""
}

MOCK_SEC_DATA = {
    "tesla": """Based on Tesla's recent SEC filings (10-K and 10-Q reports):

1. Financial Performance:
   - Revenue growth of 19% year-over-year in 2023
   - Automotive gross margin of 25.6% in Q4 2023, down from 30.6% in Q4 2022
   - Operating income of $7.9 billion in Q4 2023
   - Free cash flow of $4.4 billion in Q4 2023

2. Risk Factors:
   - Increasing competition in the EV market
   - Regulatory challenges in multiple markets
   - Supply chain constraints for critical components
   - Potential impact of economic slowdown on luxury vehicle sales

3. Forward Guidance:
   - Tesla expects to achieve 50% average annual growth in vehicle deliveries over a multi-year horizon
   - Continued investment in AI and autonomous driving technology
   - Expansion of manufacturing capacity globally
   - Growing focus on energy generation and storage business

4. Financial Ratios:
   - P/E ratio: approximately 50x (higher than traditional automakers)
   - Debt-to-equity ratio: 0.10 (relatively low debt)
   - Return on equity: 17.4%
   - Current ratio: 1.73 (healthy liquidity)""",
    
    "financial prospects": """Analysis of Tesla's financial prospects based on SEC filings and financial metrics:

1. Revenue Growth Trajectory:
   - Historical CAGR of 50% over the past 5 years
   - Analyst consensus estimates project 15-20% revenue growth for the next fiscal year
   - Energy business expected to grow at 40-50% annually

2. Profitability Outlook:
   - Automotive gross margins expected to stabilize around 25-27%
   - Operating expenses as percentage of revenue projected to decrease
   - EPS growth estimated at 10-15% for the next fiscal year

3. Balance Sheet Strength:
   - $29.1 billion in cash and cash equivalents
   - Low debt-to-equity ratio of 0.10
   - Strong free cash flow generation enabling continued R&D investment

4. Capital Allocation:
   - Significant capital expenditures planned for factory expansion
   - Continued investment in battery technology and manufacturing
   - No dividend payments anticipated in the near term

5. Regulatory Considerations:
   - Regulatory credits revenue expected to decrease as competitors produce more EVs
   - Potential impact of changing EV incentives in key markets

6. Competitive Position:
   - Leading market share in EV segment but facing increasing competition
   - Technology advantage in battery efficiency and autonomous driving
   - Brand strength continues to command premium pricing"""
}

# Initialize the chat model
# The ASI API key should be set in your environment variables as ASI_API_KEY
chat = ChatASI(model_name="asi1-mini")

# Mock search function
def mock_search(query: str) -> str:
    """Mock search function that returns predefined data based on the query."""
    query = query.lower()
    
    if "tesla" in query:
        return MOCK_SEARCH_DATA["tesla"]
    elif "financial" in query or "prospects" in query:
        return MOCK_SEARCH_DATA["financial prospects"]
    else:
        return "No relevant information found for this query."

# Mock SEC analyst function
def mock_sec_analysis(query: str) -> str:
    """Mock SEC analyst function that returns predefined financial analysis based on the query."""
    query = query.lower()
    
    if "tesla" in query:
        return MOCK_SEC_DATA["tesla"]
    elif "financial" in query or "prospects" in query:
        return MOCK_SEC_DATA["financial prospects"]
    else:
        return "No relevant SEC filings or financial analysis found for this query."

# Define the agent nodes
def search_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Search node that looks up information online."""
    # Get the last message
    last_message = state["messages"][-1]
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_template(
        """You are a search expert who can find information on the internet.
        
        Please search for information related to: {query}
        
        Provide a detailed response with the most relevant information."""
    )
    
    # Get the query from the last message
    query = state["messages"][0].content  # Use the original question
    
    # Perform the mock search
    search_results = mock_search(query)
    
    # Create a system message with the search results
    system_message = SystemMessage(content=f"Search results: {search_results}")
    
    # Create the chain
    chain = prompt | chat
    
    # Invoke the chain
    response = chain.invoke({"query": query})
    
    # Update the state
    return {"messages": state["messages"] + [system_message, response]}

def sec_analyst_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """SEC analyst node that analyzes financial documents."""
    # Get the query from the original message
    query = state["messages"][0].content  # Use the original question
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_template(
        """You are a financial analyst specializing in SEC filings and financial documents.
        
        Please analyze the financial information related to: {query}
        
        Provide a detailed analysis based on the latest SEC filings and financial reports."""
    )
    
    # Perform the mock SEC analysis
    sec_analysis = mock_sec_analysis(query)
    
    # Create a system message with the SEC analysis
    system_message = SystemMessage(content=f"SEC analysis: {sec_analysis}")
    
    # Create the chain
    chain = prompt | chat
    
    # Invoke the chain
    response = chain.invoke({"query": query})
    
    # Update the state
    return {"messages": state["messages"] + [system_message, response]}

def supervisor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Supervisor node that coordinates the research team."""
    # Create the prompt template
    prompt = ChatPromptTemplate.from_template(
        """You are the supervisor of a research team. Your job is to:
1. Understand the user's question
2. Determine which team member should handle the query next
3. Provide reasoning for your decision
4. When all necessary information is gathered, synthesize a final response

Available team members:
- Search: Good for general information lookup on the internet
- SECAnalyst: Specialized in financial documents and SEC filings

Question: {question}

Current conversation:
{conversation}

Information needed: {information_needed}
Current reasoning: {reasoning}

Your response must be in the following format:

Thinking: Your step-by-step reasoning process
Next: [Search/SECAnalyst/FINISH]
Reasoning: Brief explanation of why you chose this next step"""
    )
    
    # Get the messages from the state
    messages = state["messages"]
    information_needed = state.get("information_needed", [])
    reasoning = state.get("reasoning", "")
    question = messages[0].content if messages else ""
    
    # Format the conversation for the prompt
    conversation = ""
    for i, msg in enumerate(messages[1:], 1):  # Skip the first message (the question)
        role = msg.type
        content = msg.content
        conversation += f"\n{i}. {role.upper()}: {content}\n"
    
    # If we've gone through at least one cycle and have responses, consider finishing
    if len(messages) > 3:  # Question + at least one response from each agent
        # Increase likelihood of finishing as conversation gets longer
        if len(messages) > 5:
            reasoning += "\nWe have gathered sufficient information. Time to synthesize a final response."
    
    # Create the chain
    chain = prompt | chat
    
    # Invoke the chain
    response = chain.invoke({
        "question": question,
        "conversation": conversation,
        "information_needed": information_needed,
        "reasoning": reasoning
    })
    
    # Parse the response
    content = response.content
    
    # Extract the next step and reasoning
    next_step = "FINISH"  # Default
    new_reasoning = ""
    
    # More robust parsing
    if "Next:" in content:
        # Try to extract the next step using regex
        import re
        next_match = re.search(r"Next:\s*([^\n]+)", content)
        if next_match:
            next_candidate = next_match.group(1).strip()
            # Normalize the next step
            if "search" in next_candidate.lower():
                next_step = "Search"
            elif "sec" in next_candidate.lower() or "analyst" in next_candidate.lower():
                next_step = "SECAnalyst"
            elif "finish" in next_candidate.lower() or "final" in next_candidate.lower():
                next_step = "FINISH"
        
        # Try to extract the reasoning
        reasoning_match = re.search(r"Reasoning:\s*([^\n]+(?:\n(?!Next:|Thinking:)[^\n]+)*)", content)
        if reasoning_match:
            new_reasoning = reasoning_match.group(1).strip()
    else:
        # If no explicit Next: tag, try to infer from content
        if "search" in content.lower() and "need more information" in content.lower():
            next_step = "Search"
        elif "financial" in content.lower() and "sec" in content.lower():
            next_step = "SECAnalyst"
        
        # Extract some reasoning if possible
        if not new_reasoning:
            # Take the last paragraph as reasoning
            paragraphs = content.split("\n\n")
            if paragraphs:
                new_reasoning = paragraphs[-1]
    
    # Update the state
    return {
        "messages": messages + [response],
        "next": next_step,
        "reasoning": new_reasoning
    }

# Build the graph
def build_graph():
    """Build the agent graph."""
    # Create the graph
    graph = StateGraph(AgentState)
    
    # Add the nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("search", search_node)
    graph.add_node("sec_analyst", sec_analyst_node)
    
    # Add the conditional edges
    graph.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {
            "Search": "search",
            "SECAnalyst": "sec_analyst",
            "FINISH": END
        }
    )
    
    # Add the edges back to the supervisor
    graph.add_edge("search", "supervisor")
    graph.add_edge("sec_analyst", "supervisor")
    
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
        "reasoning": "",
        "information_needed": ["General information", "Financial data"]
    }
    
    # Run the graph with a recursion limit
    try:
        result = graph.invoke(initial_state, {"recursion_limit": 10})
        return result
    except Exception as e:
        print(f"Error: {e}")
        # Return the partial state if available
        return initial_state

# Example usage
if __name__ == "__main__":
    # Example question
    question = "What are the financial prospects for Tesla in the next year?"
    
    # Run the graph
    result = run_graph(question)
    
    # Print the conversation
    print("\nConversation:")
    for i, message in enumerate(result["messages"]):
        role = message.type
        content = message.content
        print(f"\n{role.upper()}: {content}")
    
    # Print the final reasoning
    print("\nFinal Reasoning:")
    print(result["reasoning"])
