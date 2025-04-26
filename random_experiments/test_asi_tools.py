from langchain_asi import ChatASI
from langchain_core.messages import HumanMessage
import json

# Initialize the model
llm = ChatASI(model_name='asi1-mini')

# Define a simple tool
tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                }
            },
            "required": ["location"]
        }
    }
}

# Test the tool calling
messages = [
    HumanMessage(content="What's the weather like in San Francisco?")
]

# Try calling with the tool
response = llm.invoke(
    messages,
    tools=[tool]
)

print("Response type:", type(response))
print("Response:", response)
print("Content:", response.content)
print("Additional kwargs:", response.additional_kwargs)
