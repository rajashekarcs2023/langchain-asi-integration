# OpenAI Compatibility Adapter for ASI

This document provides detailed information about the OpenAI compatibility adapter for ASI, which allows you to use ASI models with LangGraph and other libraries that expect OpenAI's interface.

## Overview

The OpenAI compatibility adapter is designed to make ASI models work seamlessly with LangGraph and other libraries that expect OpenAI's interface. It handles the differences between ASI's API and OpenAI's API, allowing you to use ASI models as drop-in replacements for OpenAI models.

## Key Features

1. **Drop-in Replacement**: Use ASI models as direct replacements for OpenAI models in LangGraph and other libraries.
2. **Tool Call Handling**: Properly formats tool calls between ASI and OpenAI formats.
3. **Structured Output**: Supports Pydantic models for structured output validation.
4. **Error Handling**: Robust error handling with fallbacks for unexpected formats.
5. **Message Formatting**: Properly formats messages between agents to help models understand the conversation flow.

## Usage

### Basic Usage

```python
from langchain_asi import ChatASI
from langchain_asi.openai_adapter import create_openai_compatible_model
import os

# Create an ASI model
asi_model = ChatASI(
    model_name="asi1-mini",
    temperature=0.7,
    max_tokens=1000,
    asi_api_key=os.getenv("ASI_API_KEY"),
    streaming=False,
    request_timeout=60.0
)

# Create an OpenAI-compatible model
openai_compatible_model = create_openai_compatible_model(asi_model)

# Use it like you would use an OpenAI model
from langchain_core.messages import HumanMessage
response = openai_compatible_model.invoke([HumanMessage(content="Hello, how are you?")])
print(response)
```

### Structured Output

The adapter supports structured output using Pydantic models:

```python
from pydantic import BaseModel, Field

class MyOutputSchema(BaseModel):
    answer: str = Field(description="The answer to the question")
    confidence: float = Field(description="Confidence score between 0 and 1")

structured_model = openai_compatible_model.with_structured_output(MyOutputSchema)
result = structured_model.invoke([HumanMessage(content="What is the capital of France?")])
print(f"Answer: {result.answer}, Confidence: {result.confidence}")
```

### With LangGraph

The adapter is designed to work seamlessly with LangGraph:

```python
from typing import Dict, List, Any, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END

# Define your state
class AgentState(TypedDict):
    messages: List[Any]

# Create an agent function
def create_agent(system_prompt):
    # Create the ASI model
    asi_model = ChatASI(
        model_name="asi1-mini",
        temperature=0.7,
        max_tokens=1000,
        asi_api_key=os.getenv("ASI_API_KEY")
    )
    
    # Create an OpenAI-compatible model
    openai_compatible_model = create_openai_compatible_model(asi_model)
    
    def agent_function(state):
        messages = state["messages"]
        full_messages = [SystemMessage(content=system_prompt)] + messages
        response = openai_compatible_model.invoke(full_messages)
        return {"messages": messages + [AIMessage(content=response)]}
    
    return agent_function

# Create your graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", create_agent("You are a helpful assistant."))
workflow.set_entry_point("agent")
workflow.add_edge("agent", END)
graph = workflow.compile()

# Run your graph
result = graph.invoke({"messages": [HumanMessage(content="Hello!")]})
```

## Advanced Usage: Multi-Agent Workflows

The adapter works well with complex multi-agent workflows in LangGraph:

```python
from pydantic import BaseModel, Field
from typing import Literal

# Define a routing schema
class RouteSchema(BaseModel):
    next: Literal["Researcher", "Writer", "FINISH"] = Field(
        description="The next agent to call or FINISH to complete the task"
    )
    reasoning: str = Field(
        description="Reasoning behind the routing decision"
    )

# Create a supervisor agent
def create_supervisor():
    asi_model = ChatASI(model_name="asi1-mini")
    openai_compatible_model = create_openai_compatible_model(asi_model)
    
    # Configure the model to return structured output
    model_with_tools = openai_compatible_model.with_structured_output(RouteSchema)
    
    def supervisor_agent(state):
        # Your supervisor logic here
        response = model_with_tools.invoke(messages)
        return {"next": response.next, "reasoning": response.reasoning}
    
    return supervisor_agent

# Create your graph with conditional routing
workflow = StateGraph(AgentState)
workflow.add_node("supervisor", create_supervisor())
workflow.add_node("researcher", create_agent("You are a researcher."))
workflow.add_node("writer", create_agent("You are a writer."))

# Add conditional edges
workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["next"],
    {
        "Researcher": "researcher",
        "Writer": "writer",
        "FINISH": END
    }
)

# Add edges from agents back to supervisor
workflow.add_edge("researcher", "supervisor")
workflow.add_edge("writer", "supervisor")
```

## Implementation Details

The OpenAI compatibility adapter works by:

1. **Message Formatting**: Converting between ASI's message format and OpenAI's message format.
2. **Tool Call Handling**: Parsing tool calls from ASI's response format and converting them to OpenAI's format.
3. **Structured Output Parsing**: Using a robust parsing mechanism with multiple fallbacks to handle different response formats.
4. **Error Handling**: Providing meaningful fallbacks when parsing fails.

## Best Practices

1. **System Messages**: Always place system messages first in your message array.
2. **Tool Definitions**: Use clear and concise tool definitions with descriptive names and parameters.
3. **Error Handling**: Always handle potential errors in your code, especially when parsing structured output.
4. **Request Parameters**: Set appropriate parameters for your ASI model:
   - `max_tokens`: Set to a reasonable value (e.g., 1000) to ensure complete responses.
   - `streaming`: Set to `False` for more reliable responses in complex workflows.
   - `request_timeout`: Set to a higher value (e.g., 60.0) for complex queries.

## Examples

Check out the examples directory for more detailed examples:

1. `examples/simple_langgraph_example.py`: A simple example of using the OpenAI compatibility adapter with LangGraph.
2. `examples/financial_analysis_example.py`: A more complex example with multiple agents and structured output.

## Troubleshooting

If you encounter issues with the adapter, try the following:

1. **Check API Key**: Ensure your ASI API key is set correctly.
2. **Increase Timeout**: For complex queries, increase the `request_timeout` parameter.
3. **Disable Streaming**: Set `streaming=False` for more reliable responses.
4. **Check Message Format**: Ensure your messages are formatted correctly, with system messages first.
5. **Simplify Tool Definitions**: If tool calls are not working, try simplifying your tool definitions.

## Limitations

1. **ASI-Specific Features**: Some ASI-specific features may not be available through the OpenAI compatibility interface.
2. **Response Format Differences**: ASI may format responses slightly differently than OpenAI, which could affect parsing in some cases.
3. **Performance**: The adapter adds a small overhead to API calls due to the additional parsing and formatting.
