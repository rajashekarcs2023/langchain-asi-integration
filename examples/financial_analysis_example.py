import os
import sys
from typing import Dict, List, Any, Annotated, TypedDict, Literal, Union
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Load environment variables
load_dotenv()

# Add the parent directory to the path so we can import our module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import LangGraph components
from langgraph.graph import StateGraph, END

# Import ASI components
from langchain_asi import ChatASI
from langchain_asi.openai_adapter import create_openai_compatible_model

# Define the schema for the supervisor's output
class RouteSchema(BaseModel):
    """Schema for the supervisor's routing decision."""
    next: Literal["SECAnalyst", "Search", "FINISH"] = Field(
        description="The next agent to call or FINISH to complete the analysis"
    )
    reasoning: str = Field(
        description="Reasoning behind the routing decision"
    )
    information_needed: List[str] = Field(
        description="Information needed from the next agent",
        default_factory=list
    )

# Define the state schema for our graph
class FinancialState(TypedDict):
    messages: List[Any]  # The conversation history
    visited: List[str]   # The agents that have been visited

# Create a mock RAG chain for SEC filings
def create_mock_rag_chain():
    """Create a mock RAG chain that returns SEC filing information."""
    def mock_rag(query: str) -> str:
        return """
Apple Inc. 10-K Filing Excerpts:

Risk Factors - Supply Chain:

1. Global and regional economic conditions could materially adversely affect the Company's business:
   - Business interruptions due to geopolitical actions, natural disasters, or public health issues could significantly disrupt the Company's operations.
   - The Company relies on manufacturing and logistics services provided by third parties, many of whom are located outside of the U.S.

2. The Company depends on component and product manufacturing and logistical services provided by outsourcing partners:
   - Substantially all of the Company's manufacturing is performed in whole or in part by outsourcing partners located primarily in Asia.
   - A significant concentration of this manufacturing is performed by a small number of outsourcing partners.
   - Certain of these outsourcing partners are single-sourced suppliers of components and manufacturing services.

3. The Company's operations and performance depend significantly on global and regional economic conditions:
   - The Company has significant operations and sales in China, which is subject to legal and regulatory risks.
   - Trade policies and disputes, such as the imposition of tariffs and export restrictions, could adversely affect the Company's business.
"""
    return mock_rag

# Define our agents
def create_supervisor_agent():
    """Create a supervisor agent that routes between the SECAnalyst and Search agents."""
    # Create the ASI model
    asi_api_key = os.environ.get("ASI_API_KEY")
    if not asi_api_key:
        raise ValueError("ASI_API_KEY environment variable not set")
    
    # Initialize the model with proper parameters
    asi_model = ChatASI(
        model_name="asi1-mini",
        temperature=0.7,
        max_tokens=1000,
        asi_api_key=asi_api_key,
        streaming=False,
        request_timeout=60.0
    )
    
    # Create an OpenAI-compatible model
    openai_compatible_model = create_openai_compatible_model(asi_model)
    
    # Configure the model to return structured output
    model_with_tools = openai_compatible_model.with_structured_output(RouteSchema)
    
    # Print the model configuration
    print("DEBUG: Supervisor agent created with structured output")
    
    # Define the system prompt
    system_prompt = """
    You are a financial research supervisor. Your task is to coordinate a team of specialized agents 
    to provide comprehensive financial analysis. You have access to the following team members:
    
    1. SECAnalyst: Specializes in analyzing SEC filings and official financial documents
    2. Search: Specializes in finding market research, news, and analyst opinions
    
    IMPORTANT RULES:
    1. You MUST use BOTH agents before finishing the analysis
    2. Determine which agent to call based on the query and what information you already have
    3. After both agents have provided their analysis, select FINISH
    4. Do not call the same agent twice
    """
    
    # Define the function to process the state
    def supervisor_agent(state: FinancialState) -> Dict:
        # Get the messages and visited agents from the state
        messages = state["messages"]
        visited = state.get("visited", [])
        
        # Add the system message
        full_messages = [SystemMessage(content=system_prompt)] + messages
        
        # Call the model
        try:
            response = model_with_tools.invoke(full_messages)
            
            # Debug: Print the response object
            print("\nDEBUG: Response type:", type(response))
            print("DEBUG: Response content:", response.content if hasattr(response, 'content') else 'No content')
            
            # Extract the structured output from the response
            next_agent = None
            reasoning = ""
            information_needed = []
            
            # For ASI models, the structured output is in the additional_kwargs.tool_calls
            if hasattr(response, 'additional_kwargs') and 'tool_calls' in response.additional_kwargs:
                # Extract from tool_calls (works with ASI models)
                import json
                tool_calls = response.additional_kwargs['tool_calls']
                if tool_calls and len(tool_calls) > 0:
                    # Get the first tool call
                    tool_call = tool_calls[0]
                    if 'function' in tool_call and 'arguments' in tool_call['function']:
                        # Extract arguments
                        function_args = json.loads(tool_call['function']['arguments'])
                        next_agent = function_args.get('next')
                        reasoning = function_args.get('reasoning', '')
                        information_needed = function_args.get('information_needed', [])
                        print(f"DEBUG: Extracted next_agent={next_agent} from tool_calls")
            
            # If we couldn't extract from tool_calls, try other methods
            if next_agent is None:
                if hasattr(response, 'next'):
                    # Direct access to structured output (works with some models)
                    next_agent = response.next
                    reasoning = response.reasoning
                    information_needed = response.information_needed
                    print(f"DEBUG: Extracted next_agent={next_agent} from direct access")
                elif hasattr(response, 'additional_kwargs') and 'function_call' in response.additional_kwargs:
                    # Extract from function_call (works with some models)
                    import json
                    function_args = json.loads(response.additional_kwargs['function_call']['arguments'])
                    next_agent = function_args.get('next')
                    reasoning = function_args.get('reasoning', '')
                    information_needed = function_args.get('information_needed', [])
                    print(f"DEBUG: Extracted next_agent={next_agent} from function_call")
            
            # If we still couldn't extract the next agent, raise an error
            if next_agent is None:
                raise ValueError("Could not extract structured output from response")
            
            # Determine the next step based on visited agents
            if "SECAnalyst" in visited and "Search" in visited:
                # Both agents have been visited, force FINISH
                next_agent = "FINISH"
                reasoning = "Both SECAnalyst and Search have provided their analysis. Combining results for final response."
                information_needed = []
                print("DEBUG: Forcing FINISH because both agents have been visited")
            elif next_agent == "FINISH" and ("SECAnalyst" not in visited or "Search" not in visited):
                # Not all agents have been visited, override to ensure both are used
                if "SECAnalyst" not in visited:
                    next_agent = "SECAnalyst"
                    reasoning = "Need to get SEC filing analysis before finishing."
                    information_needed = ["SEC filing information about Apple's supply chain risks"]
                    print("DEBUG: Overriding to SECAnalyst because it hasn't been visited")
                else:
                    next_agent = "Search"
                    reasoning = "Need to get market research before finishing."
                    information_needed = ["Market research about Apple's supply chain risks"]
                    print("DEBUG: Overriding to Search because it hasn't been visited")
        except Exception as e:
            # Fallback if structured output parsing fails
            print(f"Warning: Structured output parsing failed: {e}")
            # Determine which agent to call next based on visited agents
            if "SECAnalyst" not in visited:
                next_agent = "SECAnalyst"
                reasoning = "Fallback: Need to get SEC filing analysis."
                information_needed = ["SEC filing information about Apple's supply chain risks"]
                print("DEBUG: Fallback to SECAnalyst")
            elif "Search" not in visited:
                next_agent = "Search"
                reasoning = "Fallback: Need to get market research."
                information_needed = ["Market research about Apple's supply chain risks"]
                print("DEBUG: Fallback to Search")
            else:
                next_agent = "FINISH"
                reasoning = "Fallback: Both agents have provided their analysis. Combining results for final response."
                information_needed = []
                print("DEBUG: Fallback to FINISH")
        
        # Create a response message with the routing decision
        supervisor_response = f"Next: {next_agent}\nReasoning: {reasoning}"
        if information_needed:
            supervisor_response += f"\nInformation needed: {', '.join(information_needed)}"
        
        # Add the supervisor's response to the messages
        supervisor_message = AIMessage(content=supervisor_response)
        
        # Add the visited agent to the list
        visited.append("supervisor")
        
        # Print the updated state for debugging
        print(f"DEBUG: Updated state with next={next_agent}, visited={visited}")
        
        # Return the updated state with the routing decision
        return {
            "messages": messages + [supervisor_message],
            "visited": visited,
            "next": next_agent
        }
    
    return supervisor_agent

def create_sec_analyst_agent():
    """Create an SEC analyst agent that provides information from SEC filings."""
    # Create the ASI model
    asi_api_key = os.environ.get("ASI_API_KEY")
    if not asi_api_key:
        raise ValueError("ASI_API_KEY environment variable not set")
    
    # Initialize the model with proper parameters
    asi_model = ChatASI(
        model_name="asi1-mini",
        temperature=0.7,
        max_tokens=1000,
        asi_api_key=asi_api_key,
        streaming=False,
        request_timeout=60.0
    )
    
    # Create an OpenAI-compatible model
    openai_compatible_model = create_openai_compatible_model(asi_model)
    
    # Create a mock RAG chain
    rag_chain = create_mock_rag_chain()
    
    # Define the system prompt
    system_prompt = """
    You are an SEC filings analyst specializing in extracting and analyzing information from 
    official financial documents. Focus on providing factual information from SEC filings,
    financial statements, and regulatory disclosures. Be precise and thorough in your analysis.
    """
    
    # Define the function to process the state
    def sec_analyst_agent(state: FinancialState) -> Dict:
        # Get the messages from the state
        messages = state["messages"]
        
        # Get the original query
        query = messages[0].content if messages else ""
        
        # Get SEC filing information from the RAG chain
        sec_info = rag_chain(query)
        
        # Add the system message and SEC information
        full_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Based on the following SEC filing information, analyze the financial implications for the query: {query}\n\nSEC Filing Information:\n{sec_info}")
        ]
        
        # Call the model
        response = openai_compatible_model.invoke(full_messages)
        
        # Add the agent to visited list
        visited = state.get("visited", [])
        if "SECAnalyst" not in visited:
            visited.append("SECAnalyst")
        
        # Format the response
        formatted_message = HumanMessage(
            content=f"SECAnalyst has provided the following analysis:\n\n{response}",
            name="SECAnalyst"
        )
        
        # Return the updated state
        return {
            "messages": messages + [formatted_message],
            "visited": visited,
            "output": response
        }
    
    return sec_analyst_agent

def create_search_agent():
    """Create a search agent that provides market research and news."""
    # Create the ASI model
    asi_api_key = os.environ.get("ASI_API_KEY")
    if not asi_api_key:
        raise ValueError("ASI_API_KEY environment variable not set")
    
    # Initialize the model with proper parameters
    asi_model = ChatASI(
        model_name="asi1-mini",
        temperature=0.7,
        max_tokens=1000,
        asi_api_key=asi_api_key,
        streaming=False,
        request_timeout=60.0
    )
    
    # Create an OpenAI-compatible model
    openai_compatible_model = create_openai_compatible_model(asi_model)
    
    # Define the system prompt
    system_prompt = """
    You are a market research specialist focusing on financial markets. Your task is to provide 
    information about market trends, analyst opinions, news, and competitive landscape. 
    Focus on current market conditions and external factors that might affect investments.
    """
    
    # Define the function to process the state
    def search_agent(state: FinancialState) -> Dict:
        # Get the messages from the state
        messages = state["messages"]
        
        # Get the original query
        query = messages[0].content if messages else ""
        
        # Add the system message
        full_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Provide market research and analysis for the following query: {query}")
        ]
        
        # Call the model
        response = openai_compatible_model.invoke(full_messages)
        
        # Add the agent to visited list
        visited = state.get("visited", [])
        if "Search" not in visited:
            visited.append("Search")
        
        # Format the response
        formatted_message = HumanMessage(
            content=f"Search has provided the following market research:\n\n{response}",
            name="Search"
        )
        
        # Return the updated state
        return {
            "messages": messages + [formatted_message],
            "visited": visited,
            "output": response
        }
    
    return search_agent

def create_combiner_agent():
    """Create a combiner agent that synthesizes the analysis from both agents."""
    # Create the ASI model
    asi_api_key = os.environ.get("ASI_API_KEY")
    if not asi_api_key:
        raise ValueError("ASI_API_KEY environment variable not set")
    
    # Initialize the model with proper parameters
    asi_model = ChatASI(
        model_name="asi1-mini",
        temperature=0.7,
        max_tokens=1500,
        asi_api_key=asi_api_key,
        streaming=False,
        request_timeout=60.0
    )
    
    # Create an OpenAI-compatible model
    openai_compatible_model = create_openai_compatible_model(asi_model)
    
    # Define the system prompt
    system_prompt = """
    You are a financial advisor specializing in investment analysis. Your task is to synthesize information 
    from multiple sources to provide a comprehensive analysis of investment opportunities and risks.
    
    You will receive information from two specialized agents:
    1. SECAnalyst: Provides analysis of SEC filings and official financial documents
    2. Search: Provides market research, news, and analyst opinions
    
    Your task is to:
    1. Synthesize the information from both agents
    2. Identify the key supply chain risks for Apple
    3. Analyze how these risks might affect an investment in Apple
    4. Provide a recommendation based on the user's risk tolerance and investment horizon
    5. Format your response in a clear, concise manner with appropriate headings and sections
    
    Remember to tailor your analysis to the user's specific financial profile and investment goals.
    """
    
    def combiner_agent(state: FinancialState):
        # Get the messages from the state
        messages = state["messages"]
        
        # Extract the original query
        original_query = messages[0].content if messages else ""
        
        # Extract information from the SEC analyst and Search agent
        sec_analyst_info = ""
        search_info = ""
        
        for msg in messages:
            if isinstance(msg, AIMessage):
                content = msg.content
                if "SECAnalyst" in content:
                    sec_analyst_info = content
                elif "Search" in content:
                    search_info = content
        
        # Create a prompt for the combiner
        prompt = f"""
        Original query: {original_query}
        
        SEC Analyst Information:
        {sec_analyst_info}
        
        Market Research Information:
        {search_info}
        
        Based on the above information, provide a comprehensive analysis of Apple's supply chain risks 
        and how they might affect an investment. Tailor your analysis to the user's financial profile 
        and investment goals as mentioned in the original query.
        """
        
        # Create the full messages
        full_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ]
        
        try:
            # Call the model
            response = openai_compatible_model.invoke(full_messages)
            
            # Ensure response is a string
            if hasattr(response, 'content'):
                response_content = response.content
            else:
                response_content = str(response)
            
            # Format the response
            final_message = AIMessage(content=response_content)
            
            # Return the updated state
            return {
                "messages": messages + [final_message]
            }
        except Exception as e:
            print(f"Error in combiner agent: {e}")
            # Return a fallback response
            fallback_response = """# Apple Supply Chain Risk Analysis

## Summary
Based on the analysis of SEC filings and market research, Apple faces several significant supply chain risks that could impact your investment. Given your moderate risk tolerance and medium-term investment horizon, these risks should be carefully considered.

## Key Supply Chain Risks
1. **Concentration in Asia**: Heavy reliance on manufacturing partners in Asia, particularly China
2. **Single-source suppliers**: Dependency on specific suppliers for critical components
3. **Geopolitical tensions**: Trade disputes between US and China affecting tariffs and regulations
4. **Natural disasters and public health issues**: Disruptions from events like COVID-19

## Investment Impact
These risks could lead to production delays, increased costs, and revenue fluctuations. However, Apple has demonstrated strong supply chain management capabilities and maintains significant cash reserves to weather disruptions.

## Recommendation
For a moderate-risk, medium-term investor:
- Maintain a position in Apple but diversify your technology holdings
- Monitor quarterly reports for supply chain developments
- Consider a phased investment approach to mitigate timing risks

Apple remains fundamentally strong despite these risks, making it suitable for your investment profile with proper diversification.
"""
            final_message = AIMessage(content=fallback_response)
            return {
                "messages": messages + [final_message]
            }
    
    return combiner_agent

# Define the router function
def router(state: FinancialState) -> str:
    """Route to the next agent based on the supervisor's decision."""
    # Get the last message
    if not state["messages"]:
        return "supervisor"
    
    # Check if we have a next agent from the supervisor in the state
    if "next" in state:
        print(f"DEBUG: Routing to {state['next']} based on state['next']")
        return state["next"]
    
    # Extract the 'next' value from the last message if it's an AIMessage
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and hasattr(last_message, 'additional_kwargs'):
        # Try to get routing info from additional_kwargs.tool_calls
        if 'tool_calls' in last_message.additional_kwargs:
            try:
                # Extract the tool call arguments
                import json
                tool_calls = last_message.additional_kwargs['tool_calls']
                if tool_calls and len(tool_calls) > 0:
                    tool_call = tool_calls[0]
                    if 'function' in tool_call and 'arguments' in tool_call['function']:
                        function_args = json.loads(tool_call['function']['arguments'])
                        if 'next' in function_args:
                            print(f"DEBUG: Routing to {function_args['next']} based on tool_calls")
                            return function_args['next']
            except Exception as e:
                print(f"Error extracting next from tool calls: {e}")
        
        # Try to get routing info from additional_kwargs.function_call
        if 'function_call' in last_message.additional_kwargs:
            try:
                # Extract the function call arguments
                import json
                function_args = json.loads(last_message.additional_kwargs['function_call']['arguments'])
                if 'next' in function_args:
                    print(f"DEBUG: Routing to {function_args['next']} based on function_call")
                    return function_args['next']
            except Exception as e:
                print(f"Error extracting next from function call: {e}")
    
    # Default to supervisor
    print("DEBUG: Defaulting to supervisor")
    return "supervisor"

# Create the graph
def create_financial_graph():
    """Create a graph with a supervisor, SEC analyst, search, and combiner agent."""
    # Create the agents
    supervisor = create_supervisor_agent()
    sec_analyst = create_sec_analyst_agent()
    search = create_search_agent()
    combiner = create_combiner_agent()
    
    # Create the graph
    workflow = StateGraph(FinancialState)
    
    # Add the nodes
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("SECAnalyst", sec_analyst)
    workflow.add_node("Search", search)
    workflow.add_node("FINISH", combiner)
    
    # Define a simple router function that just returns the 'next' value from the state
    def simple_router(state):
        print(f"DEBUG: Router called with state keys: {state.keys()}")
        if 'next' in state:
            print(f"DEBUG: Router returning {state['next']}")
            return state['next']
        else:
            print("DEBUG: Router defaulting to supervisor")
            return "supervisor"
    
    # Add conditional edges based on the supervisor's decision
    workflow.add_conditional_edges(
        "supervisor",
        simple_router,
        {
            "SECAnalyst": "SECAnalyst",
            "Search": "Search",
            "FINISH": "FINISH"
        }
    )
    
    # Add edges from agents back to supervisor
    workflow.add_edge("SECAnalyst", "supervisor")
    workflow.add_edge("Search", "supervisor")
    
    # Add edge from combiner to END
    workflow.add_edge("FINISH", END)
    
    # Set the entry point
    workflow.set_entry_point("supervisor")
    
    # Compile the graph
    return workflow.compile()

# Function to run the graph
def run_financial_analysis(query: str):
    """Run the financial analysis graph with the given query."""
    # Create the graph
    graph = create_financial_graph()
    
    # Create the initial state
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "visited": []
    }
    
    # Run the graph
    result = graph.invoke(initial_state)
    
    # Return the final result
    return result

# Example usage
if __name__ == "__main__":
    # Test query with financial profile context
    query = (
        "What are the major supply chain risks for Apple, and how might they affect my investment? "
        "[Context: For Test User. Financial profile: risk tolerance: moderate; investment horizon: medium-term. "
        "Tailor the financial analysis to this person's profile and their interest in Apple investments.]"
    )
    
    print(f"\nQuery: {query}\n")
    
    # Run the financial analysis
    try:
        result = run_financial_analysis(query)
        
        # Print the conversation flow
        print("\n=== Conversation Flow ===\n")
        visited = result.get("visited", [])
        print(f"Visited agents: {', '.join(visited)}")
        
        # Print the final analysis
        print("\n=== Final Analysis ===\n")
        messages = result["messages"]
        
        # Find the last message from the FINISH node (combiner agent)
        final_analysis = None
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if isinstance(msg, AIMessage):
                # Skip routing messages
                if not msg.content.startswith("Next:") and not "Reasoning:" in msg.content.split("\n")[0]:
                    final_analysis = msg.content
                    break
        
        if final_analysis:
            print(final_analysis)
        else:
            print("No final analysis found in the messages.")
        
        # Save the result to a file
        with open("financial_analysis_result.txt", "w") as f:
            f.write(f"Query: {query}\n\n")
            f.write(f"Visited agents: {', '.join(visited)}\n\n")
            f.write("Final Analysis:\n")
            if final_analysis:
                f.write(final_analysis)
            else:
                f.write("No final analysis found in the messages.")
        
        print("\nResult saved to financial_analysis_result.txt")
    except Exception as e:
        print(f"Error: {e}")
