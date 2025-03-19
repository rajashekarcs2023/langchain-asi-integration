import os
import sys
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_asi import ASI1ChatModel

# Load environment variables
load_dotenv()

# Check if ASI1_API_KEY is set
if not os.environ.get("ASI1_API_KEY"):
    print("Error: ASI1_API_KEY environment variable not found.")
    print("Please create a .env file with your ASI1_API_KEY or set it directly in your environment.")
    print("Example .env file content: ASI1_API_KEY=your-api-key-here")
    sys.exit(1)

# Initialize our ASI1 model
llm = ASI1ChatModel(
    model_name="asi1-mini",
    temperature=0.3,
    max_tokens=4000,
    api_key=os.environ.get("ASI1_API_KEY")
)

# Create a memory for conversation context
memory = ConversationBufferMemory(return_messages=True)

# Create the risk assessment prompt
risk_prompt = ChatPromptTemplate.from_template("""Analyze the user's risk tolerance based on their financial question:
    
User question: {question}
    
Provide a risk assessment on a scale of 1-10, where:
1 = Extremely risk-averse
10 = Extremely risk-tolerant
    
Risk assessment:""")

# Create the financial advice prompt
advice_prompt = ChatPromptTemplate.from_template("""You are a financial advisor. Based on the user's question and their risk assessment, provide personalized financial advice.
    
User question: {question}
Risk assessment: {risk_assessment}
    
Provide specific, actionable financial advice that matches their risk tolerance. Include:
1. Short-term recommendations
2. Medium-term strategy
3. Long-term outlook
    
Financial advice:""")

# Create the response prompt
response_prompt = ChatPromptTemplate.from_template("""You are a helpful financial assistant.
    
Chat history: {history}
    
User question: {question}
Risk assessment: {risk_assessment}
Financial advice: {financial_advice}
    
Provide a friendly, conversational response to the user that incorporates the financial advice while maintaining a natural tone.
    
Response:""")

# Create a simple financial advisor chain
def advisor_chain():
    # Step 1: Get risk assessment
    def get_risk_assessment(inputs):
        question = inputs["question"]
        risk_result = llm.invoke(risk_prompt.format(question=question))
        return {"risk_assessment": risk_result.content, "question": question}
    
    # Step 2: Get financial advice
    def get_financial_advice(inputs):
        question = inputs["question"]
        risk_assessment = inputs["risk_assessment"]
        advice_result = llm.invoke(advice_prompt.format(
            question=question, 
            risk_assessment=risk_assessment
        ))
        return {
            "financial_advice": advice_result.content,
            "risk_assessment": risk_assessment,
            "question": question
        }
    
    # Step 3: Format response
    def format_response(inputs):
        question = inputs["question"]
        risk_assessment = inputs["risk_assessment"]
        financial_advice = inputs["financial_advice"]
        history = memory.load_memory_variables({})
        history = history.get("history", "")
        
        response_result = llm.invoke(response_prompt.format(
            question=question,
            risk_assessment=risk_assessment,
            financial_advice=financial_advice,
            history=history
        ))
        
        # Save to memory
        memory.save_context({"input": question}, {"output": response_result.content})
        
        return {
            "response": response_result.content,
            "risk_assessment": risk_assessment,
            "financial_advice": financial_advice
        }
    
    def process_question(inputs):
        # Run all steps in sequence
        risk_result = get_risk_assessment(inputs)
        advice_result = get_financial_advice(risk_result)
        final_result = format_response(advice_result)
        return final_result
    
    return process_question

def main():
    print("ASI1-Powered Financial Advisor")
    print("------------------------------")
    print("Ask me any financial question, or type 'exit' to quit.")
    print("Example questions:")
    print("- Should I invest in high-growth tech stocks?")
    print("- How should I save for retirement?")
    print("- What's a good way to build an emergency fund?")
    print()
    
    # Create the chain
    chain = advisor_chain()
    
    while True:
        question = input("You: ")
        if question.lower() == 'exit':
            break
            
        try:
            # Run the chain
            result = chain({"question": question})
            
            # Print the response
            print("\nFinancial Advisor:", result["response"])
            
            # Optionally show debugging info
            print("\n--- Debug Info ---")
            print(f"Risk Assessment: {result['risk_assessment']}")
            print(f"Financial Advice:\n{result['financial_advice']}")
            print("------------------")
            
        except Exception as e:
            print(f"\nError: {str(e)}")
        
        print()

if __name__ == "__main__":
    main()
