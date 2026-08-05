from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()  # Load environment variables from .env file

embeddings = OpenAIEmbeddings(model = 'text-embedding-3-large', dimensions = 300)

documents = [
    "Delhi is the capital of India",
    "Paris is the capital of France",
    "London is the capital of the United Kingdom",
    "Virat Kohli is a famous cricketer",
    "Sachin Tendulkar is a legendary cricketer",
]

query = "Who is the best cricketer in the world?"

query_embedding = embeddings.embed_query(query)
document_embeddings = embeddings.embed_documents(documents)

similarities = cosine_similarity([query_embedding], document_embeddings)

index, score = sorted(enumerate(similarities[0]), key=lambda x: x[1])[-1]

print(query)
print(documents[index])
print(f"Similarity score: {score}")