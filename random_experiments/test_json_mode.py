from langchain_asi import ChatASI

# Create a simple JSON schema
schema = {
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

# Initialize the model
llm = ChatASI(model_name='asi1-mini')

# Create a structured output chain with JSON mode
structured_llm = llm.with_structured_output(
    schema=schema,
    method="json_mode"
)

# Test with a simple query
result = structured_llm.invoke(
    "Who should analyze Apple's financial risks? Choose between Search and SECAnalyst."
)

print(result)
