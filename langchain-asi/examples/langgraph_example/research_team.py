"""Research team example using LangGraph and ASI1ChatModel."""
import os
from typing import Annotated, Dict, List, TypedDict

import operator
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool
import functools

from langgraph.graph import StateGraph, END
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_asi import ASI1ChatModel
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.runnables import RunnablePassthrough


# Set your API key - for a real implementation, use environment variables
os.environ["ASI1_API_KEY"] = "your-apikey"  # Replace with your actual API key


class ResearchTeamState(TypedDict):
    """Define the state structure for the research team."""
    messages: Annotated[List[BaseMessage], operator.add]
    next: str


def agent_node(state: Dict, agent: AgentExecutor, name: str) -> Dict:
    """Process a state with an agent and return the updated state."""
    # Get the last message
    last_message = state["messages"][-1]
    
    # Run the agent
    result = agent.invoke({"input": last_message.content})
    
    # Create a new message with the agent's response
    response = HumanMessage(content=f"[{name}]: {result['output']}")
    
    # Return updated state
    return {"messages": state["messages"] + [response]}


def create_search_agent(llm):
    """Create a search agent."""
    # Define search tool
    search_tool = Tool(
        name="search",
        description="Search for information on the internet",
        func=lambda query: f"Search results for: {query}\n- Result 1: Information about {query}\n- Result 2: More details about {query}\n- Result 3: Additional context for {query}"
    )
    
    # Define the prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a search expert. Use the search tool to find information.\n\nTools: {tools}\n\nUse the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [{tool_names}]\nAction Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question"),
        ("human", "{input}"),
        ("human", "{agent_scratchpad}"),
    ])
    
    # Create the agent
    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
            tools=lambda _: search_tool.description,
            tool_names=lambda _: search_tool.name
        )
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )
    
    # Create the executor
    return AgentExecutor(agent=agent, tools=[search_tool], verbose=True, handle_parsing_errors=True)


def create_analyst_agent(llm):
    """Create a financial analyst agent."""
    # Define analysis tool
    analysis_tool = Tool(
        name="analyze",
        description="Analyze financial data",
        func=lambda query: f"Financial analysis for: {query}\n- Revenue: $10M\n- Profit: $2M\n- Growth: 15%\n- Risk factors: Market volatility, competition"
    )
    
    # Define the prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a financial analyst. Use the analyze tool to provide financial insights.\n\nTools: {tools}\n\nUse the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [{tool_names}]\nAction Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question"),
        ("human", "{input}"),
        ("human", "{agent_scratchpad}"),
    ])
    
    # Create the agent
    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
            tools=lambda _: analysis_tool.description,
            tool_names=lambda _: analysis_tool.name
        )
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )
    
    # Create the executor
    return AgentExecutor(agent=agent, tools=[analysis_tool], verbose=True, handle_parsing_errors=True)


def create_supervisor(llm):
    """Create a supervisor function that determines the next step."""
    # Define the prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a supervisor coordinating a research team. 
        Based on the query and any previous responses, decide who should act next:
        - 'search': For general information lookup
        - 'analyst': For financial analysis
        - 'FINISH': When enough information has been gathered
        
        Respond with ONLY ONE of these options."""),
        ("human", "{query}"),
    ])
    
    def route(state):
        """Route to the next agent or finish."""
        # Get the messages
        messages = state["messages"]
        last_message = messages[-1]
        
        # If this is the first message, start with search
        if len(messages) == 1 and isinstance(last_message, HumanMessage):
            return {"next": "search"}
        
        # If we've already gone through both agents, finish
        if len(messages) >= 3:
            return {"next": "FINISH"}
        
        # Otherwise, ask the LLM to decide
        response = llm.invoke(prompt.invoke({"query": last_message.content}))
        
        # Parse the response to get the next agent
        content = response.content.strip().lower()
        
        if "search" in content:
            next_agent = "search"
        elif "analyst" in content:
            next_agent = "analyst"
        else:
            next_agent = "FINISH"
            
        return {"next": next_agent}
    
    return route


def create_research_graph():
    """Create the research team graph."""
    # Initialize the LLM
    llm = ASI1ChatModel(
        model_name="asi1-mini",
        temperature=0.7,
    )
    
    # Create the agents
    search_agent = create_search_agent(llm)
    analyst_agent = create_analyst_agent(llm)
    supervisor = create_supervisor(llm)
    
    # Create the agent nodes
    search_node = functools.partial(agent_node, agent=search_agent, name="Search")
    analyst_node = functools.partial(agent_node, agent=analyst_agent, name="Analyst")
    
    # Create the graph
    graph = StateGraph(ResearchTeamState)
    
    # Add nodes
    graph.add_node("search", search_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("supervisor", supervisor)
    
    # Add edges
    graph.add_edge("search", "supervisor")
    graph.add_edge("analyst", "supervisor")
    
    # Add conditional edges from supervisor
    graph.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {
            "search": "search",
            "analyst": "analyst",
            "FINISH": END
        },
    )
    
    # Set entry point
    graph.set_entry_point("supervisor")
    
    return graph.compile()


def main():
    """Run the example."""
    # Create the graph
    graph = create_research_graph()
    
    # Define the query
    query = "What are the recent financial performance and risk factors for Tesla?"
    
    # Run the graph
    result = graph.invoke({"messages": [HumanMessage(content=query)]})
    
    # Print the result
    print("\nFinal Result:\n")
    for message in result["messages"]:
        print(f"{message.content}\n")


if __name__ == "__main__":
    main()
