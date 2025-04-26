from langchain_asi import ChatASI
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description='The person name')
    age: int = Field(description='The person age')

llm = ChatASI(model_name='asi1-mini')
structured_llm = llm.with_structured_output(Person)

result = structured_llm.invoke('Create a person named John who is 30 years old')
print(result)
