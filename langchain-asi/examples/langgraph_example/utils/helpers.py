"""Helper functions for the research team."""
from typing import Any, Dict, List

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


def agent_node(state: Dict[str, Any], agent: Any, name: str) -> Dict[str, Any]:
    """Process a state with an agent and return the updated state."""
    # Get the last message if it exists, otherwise use all messages
    messages = state["messages"]
    last_message = messages[-1] if messages else None
    
    # If the last message is from a human or doesn't exist, use all messages
    if not last_message or isinstance(last_message, HumanMessage):
        agent_input = {"messages": messages}
    else:
        # Create a new human message with the content from the last AI message
        new_message = HumanMessage(content=last_message.content)
        agent_input = {"messages": [new_message]}
    
    # Run the agent
    result = agent.invoke(agent_input)
    
    # Extract the response message
    if isinstance(result, dict) and "messages" in result:
        response = result["messages"][-1]
    elif isinstance(result, str):
        response = AIMessage(content=result)
    else:
        response = AIMessage(content=str(result))
    
    # Add agent name to response for clarity
    response_content = f"[{name}]: {response.content}"
    agent_response = AIMessage(content=response_content)
    
    # Return updated state
    return {"messages": messages + [agent_response]}
