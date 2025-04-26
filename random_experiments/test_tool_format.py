from langchain_asi import ChatASI
from langchain_core.messages import HumanMessage
import json

# Initialize the model
llm = ChatASI(model_name='asi1-mini')

# Define a tool in the OpenAI format
openai_tool = {
    "type": "function",
    "function": {
        "name": "route",
        "description": "Select the next role based on query analysis.",
        "parameters": {
            "type": "object",
            "properties": {
                "next": {
                    "type": "string",
                    "enum": ["FINISH", "Search", "SECAnalyst"],
                    "description": "The next agent to act"
                }
            },
            "required": ["next"]
        }
    }
}

# Define a simpler tool format
simple_tool = {
    "name": "route",
    "description": "Select the next role based on query analysis.",
    "parameters": {
        "type": "object",
        "properties": {
            "next": {
                "type": "string",
                "enum": ["FINISH", "Search", "SECAnalyst"],
                "description": "The next agent to act"
            }
        },
        "required": ["next"]
    }
}

# Test with OpenAI format
print("Testing with OpenAI format:")
try:
    response1 = llm.invoke(
        [HumanMessage(content="Who should analyze Apple's financial risks?")],
        tools=[openai_tool],
        tool_choice={"type": "function", "function": {"name": "route"}}
    )
    print("Response:", response1)
    print("Additional kwargs:", response1.additional_kwargs)
except Exception as e:
    print("Error with OpenAI format:", e)

# Test with simpler format
print("\nTesting with simpler format:")
try:
    response2 = llm.invoke(
        [HumanMessage(content="Who should analyze Apple's financial risks?")],
        tools=[simple_tool],
        tool_choice="route"
    )
    print("Response:", response2)
    print("Additional kwargs:", response2.additional_kwargs)
except Exception as e:
    print("Error with simpler format:", e)

# Test with bind_tools
print("\nTesting with bind_tools (OpenAI format):")
try:
    llm_with_tools1 = llm.bind_tools(tools=[openai_tool], tool_choice={"type": "function", "function": {"name": "route"}})
    response3 = llm_with_tools1.invoke([HumanMessage(content="Who should analyze Apple's financial risks?")])
    print("Response:", response3)
    print("Additional kwargs:", response3.additional_kwargs)
except Exception as e:
    print("Error with bind_tools (OpenAI format):", e)

# Test with bind_tools simpler format
print("\nTesting with bind_tools (simpler format):")
try:
    llm_with_tools2 = llm.bind_tools(tools=[simple_tool], tool_choice="route")
    response4 = llm_with_tools2.invoke([HumanMessage(content="Who should analyze Apple's financial risks?")])
    print("Response:", response4)
    print("Additional kwargs:", response4.additional_kwargs)
except Exception as e:
    print("Error with bind_tools (simpler format):", e)
