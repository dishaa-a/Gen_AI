from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

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

parser = JsonOutputParser()

template = PromptTemplate(
    template = "Give me the name, age and city of a fictional person \n {format_instructions}",
    input_variables = [],
    partial_variables = {'format_instructions': parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({})

print(result)