from langchain_asi import ChatASI
from langchain_core.messages import HumanMessage
import json

# Initialize the model
llm = ChatASI(model_name='asi1-mini')

# Define a simple tool
tool = {
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

# Test the tool calling directly
response = llm.invoke(
    [HumanMessage(content="Who should analyze Apple's financial risks?")],
    tools=[tool],
    tool_choice={"type": "function", "function": {"name": "route"}}
)

print("Response:", response)
print("Content:", response.content)
print("Additional kwargs:", response.additional_kwargs)

# Now test with bind_tools
llm_with_tools = llm.bind_tools(tools=[tool], tool_choice={"type": "function", "function": {"name": "route"}})
response2 = llm_with_tools.invoke([HumanMessage(content="Who should analyze Apple's financial risks?")])

print("\nResponse with bind_tools:", response2)
print("Content:", response2.content)
print("Additional kwargs:", response2.additional_kwargs)
