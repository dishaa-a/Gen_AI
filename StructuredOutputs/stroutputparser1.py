from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id = "Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

# 1st Prompt -> detailed report
template1 = PromptTemplate(
    template = "Write a detailed report on {topic}.",
    input_variables = ["topic"]
)

# 2nd Prompt  -> Summary of the report
template2 = PromptTemplate(
    template = "Write a 5 line summary of the following report: {report}.",
    input_variables = ["report"]
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

chain.invoke({"topic": "Artificial Intelligence in Healthcare"})

result = chain.invoke({"topic": "Artificial Intelligence in Healthcare"})
print(result)