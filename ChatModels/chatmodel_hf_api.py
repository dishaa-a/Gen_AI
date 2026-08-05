from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

llm = HuggingFaceEndpoint(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0:featherless-ai",
    task = "text-generation"
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of France?")

print(result.content)