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
            },
            "reasoning": {
                "type": "string",
                "description": "Explanation for why this agent should act next"
            }
        },
        "required": ["next", "reasoning"]
    }
}

# Test direct tool calling
print("Testing direct tool calling:")
try:
    response = llm.invoke(
        [HumanMessage(content="Who should analyze Apple's financial risks?")],
        tools=[tool],
        tool_choice="route"
    )
    print("Response:", response)
    print("Content:", response.content)
    print("Additional kwargs:", response.additional_kwargs)
except Exception as e:
    print("Error with direct tool calling:", e)

# Test with bind_tools
print("\nTesting with bind_tools:")
try:
    llm_with_tools = llm.bind_tools(tools=[tool], tool_choice="route")
    response2 = llm_with_tools.invoke([HumanMessage(content="Who should analyze Apple's financial risks?")])
    print("Response:", response2)
    print("Content:", response2.content)
    print("Additional kwargs:", response2.additional_kwargs)
except Exception as e:
    print("Error with bind_tools:", e)

# Test the response parsing
print("\nTesting response parsing:")
try:
    from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
    parser = JsonOutputFunctionsParser()
    if hasattr(response, 'additional_kwargs') and 'tool_calls' in response.additional_kwargs:
        print("Tool calls found in response:", response.additional_kwargs['tool_calls'])
        try:
            parsed = parser.parse([response])
            print("Parsed result:", parsed)
        except Exception as e:
            print("Error parsing with JsonOutputFunctionsParser:", e)
    else:
        print("No tool_calls found in response.additional_kwargs")
except Exception as e:
    print("Error testing response parsing:", e)
