from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema

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

schema = [
    ResponseSchema(name="name", description="The name of the person"),
    ResponseSchema(name="age", description="The age of the person"),
    ResponseSchema(name="city", description="The city where the person lives")
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template = "Give me the name, age and city of a fictional person \n {format_instructions}",
    input_variables = [],
    partial_variables = {'format_instructions': parser.get_format_instructions()}
)  

chain = template | model | parser

result = chain.invoke({})

print(result)

# Disadvantage of StructuredOutputParser: It is more complex to set up and requires defining a schema, which may not be necessary for simpler tasks.
# Data validation: StructuredOutputParser cannot validate the output against the defined schema, ensuring that the output adheres to the expected structure and types. 
# So we use Pydantic output parser for data validation. It is a more robust solution for ensuring that the output meets the expected structure and types, especially when dealing with complex data structures or when strict validation is required.
