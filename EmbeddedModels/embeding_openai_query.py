from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

embeddings = OpenAIEmbeddings(model = 'text-embedding-3-large', dimensions = 32)

result = embeddings.embed_query("Delhi is the capital of India")

print(str(result))