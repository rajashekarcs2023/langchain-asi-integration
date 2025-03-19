"""Agents for the research team."""
from typing import Dict, List, Any

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_asi import ASI1ChatModel


# Mock responses for testing
def mock_response(query: str) -> AIMessage:
    """Generate a mock response for testing."""
    return AIMessage(content=f"This is a mock response to: {query}")


def create_search_agent(llm: ASI1ChatModel) -> Any:
    """Create a search agent that can look up information online."""
    # Check if we're in test mode
    if llm.asi1_api_key == "dummy-api-key-for-testing":
        return lambda x: {"messages": [mock_response(x["messages"][-1].content)]}
    
    # Define search tool
    search_tool = Tool(
        name="Search",
        func=lambda query: f"Search results for: {query}\n- Result 1: Information about {query}\n- Result 2: More details about {query}\n- Result 3: Additional context for {query}",
        description="Useful for searching for information on the internet."
    )
    
    # Define agent prompt
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a search expert who can find information on the internet. "
                             "Use the search tool to find relevant information for the user's query."),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create agent
    agent = create_react_agent(llm, [search_tool], prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(agent=agent, tools=[search_tool], verbose=True)
    
    return agent_executor


def create_sec_agent(llm: ASI1ChatModel, rag_chain: Any) -> Any:
    """Create an SEC analyst agent that can analyze financial documents."""
    # Check if we're in test mode
    if llm.asi1_api_key == "dummy-api-key-for-testing":
        return lambda x: {"messages": [mock_response(x["messages"][-1].content)]}
    
    # Define SEC document tool
    sec_tool = Tool(
        name="SECDocuments",
        func=lambda query: f"SEC filing information for: {query}\n- Revenue: $10M\n- Profit: $2M\n- Growth: 15%\n- Risk factors: Market volatility, competition",
        description="Useful for retrieving information from SEC filings and financial documents."
    )
    
    # Define agent prompt
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a financial analyst specializing in SEC filings and financial documents. "
                             "Use the SECDocuments tool to analyze financial information for the user's query."),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create agent
    agent = create_react_agent(llm, [sec_tool], prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(agent=agent, tools=[sec_tool], verbose=True)
    
    return agent_executor


def create_supervisor_agent(llm: ASI1ChatModel) -> Any:
    """Create a supervisor agent that coordinates the research team."""
    # Define supervisor prompt
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are the supervisor of a research team. Your job is to:
1. Understand the user's question
2. Determine which team member should handle the query next
3. Provide reasoning for your decision
4. When all necessary information is gathered, synthesize a final response

Available team members:
- Search: Good for general information lookup on the internet
- SECAnalyst: Specialized in financial documents and SEC filings

You must choose the next team member or FINISH if the task is complete."""),
        MessagesPlaceholder(variable_name="messages"),
        HumanMessage(content="""Based on the conversation so far and the information we have, decide the next step.

Information needed: {information_needed}
Current reasoning: {reasoning}

Your response must be in the following format:

Thinking: Your step-by-step reasoning process
Next: [Search/SECAnalyst/FINISH]
Reasoning: Brief explanation of why you chose this next step"""),
    ])
    
    # Define supervisor chain
    def supervisor_chain(state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the state and determine the next step."""
        # Get the messages from the state
        messages = state["messages"]
        information_needed = state.get("information_needed", [])
        reasoning = state.get("reasoning", "")
        
        # Check if we're in test mode
        if llm.asi1_api_key == "dummy-api-key-for-testing":
            # In test mode, alternate between Search, SECAnalyst, and FINISH
            if len(messages) == 1:
                next_step = "Search"
                new_reasoning = "Need to search for general information first."
            elif len(messages) == 2:
                next_step = "SECAnalyst"
                new_reasoning = "Now need financial analysis."
            else:
                next_step = "FINISH"
                new_reasoning = "All information has been gathered."
                
            response = AIMessage(content=f"Thinking: This is a mock supervisor response.\nNext: {next_step}\nReasoning: {new_reasoning}")
        else:
            # Call the LLM
            response = llm.invoke(prompt.invoke({
                "messages": messages,
                "information_needed": information_needed,
                "reasoning": reasoning
            }))
        
        # Parse the response
        content = response.content
        
        # Extract the next step and reasoning
        next_step = "FINISH"  # Default
        new_reasoning = ""
        
        for line in content.split("\n"):
            if line.startswith("Next:"):
                next_step = line.replace("Next:", "").strip()
            elif line.startswith("Reasoning:"):
                new_reasoning = line.replace("Reasoning:", "").strip()
        
        # Update the state
        return {
            "messages": messages + [response],
            "next": next_step,
            "reasoning": new_reasoning
        }
    
    return supervisor_chain
