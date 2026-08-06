# Structured Output Parser in langchain that uses Pydantic models to enforce schema validation when processing LLM responses.

# Why use PydanticOutputParser instead of StructuredOutputParser?
# Strict Schema Enforcement: PydanticOutputParser uses Pydantic models to define the expected structure of the output. 
# Type Safety: Automatically converts LLM outputs into python objects.
# Easy Validation: Uses Pydantic's built-in validation to ensure that the output adheres to the defined schema, raising errors if the output does not match the expected structure or types.
# Seamless Integration: Works well with other Pydantic-based systems and libraries, making it easier to integrate into existing workflows that rely on Pydantic models.

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id = "Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation",
     pipeline_kwargs={
        "max_new_tokens": 256,
        "return_full_text": False,   # only return the new reply, not the whole prompt
    }
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):

    name: str = Field(description='Name of the person')
    age: int = Field(description='Age of the person')
    city: str = Field(description='Name of the city the person belongs to')

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Generate the name, age and city of a fictional {place} person \n {format_instructions}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'place': 'srilankan'})

print(result)