from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

documents = [
    "Delhi is the capital of India",
    "Paris is the capital of France",
    "London is the capital of the United Kingdom"

]

embeddings = OpenAIEmbeddings(model = 'text-embedding-3-large', dimensions = 32)

result = embeddings.embed_documents(documents)

for i, doc_result in enumerate(result):
    print(f"Document {i + 1}: {str(doc_result)}")