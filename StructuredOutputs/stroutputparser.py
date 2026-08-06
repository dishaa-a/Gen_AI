from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id = "Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 512,
        "temperature": 0.5,
        "do_sample": True,
        "return_full_text": False,   # <-- key fix: only return the new reply
    }
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

prompt1 = template1.invoke

result = model.invoke(prompt1)

prompt2 = template2.invoke

result1 = model.invoke(prompt2)

print(result1)