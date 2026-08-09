# Imports Hugging Face models, embeddings, FAISS, document loader, and text splitter.
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load the document: Loads the content of docs.txt into LangChain Document objects.
loader = TextLoader("docs.txt")  
documents =loader.load()

# Split the text into smaller chunks: Breaks the document into smaller overlapping chunks so they can be efficiently searched.
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Convert text into embeddings & store in FAISS: Converts the text chunks into numerical vectors (embeddings) and stores them in a FAISS vector database.
vectorestore = FAISS.from_documents(docs, HuggingFaceEmbeddings)

# Create a retriever (fetches relevant documents): Creates a component that searches the vector database for chunks relevant to a query.
retriever = vectorestore.as_retriever()

# Manually Retrieve Relevant Documents
query = "What are the key takeaways from the document?"   # Defines the question that we want the model to answer.
retrieved_docs = retriever._get_relevant_documents(query)   # Searches FAISS and retrieves the chunks that are most relevant to the question.

# Combine Retrieved Text into a Single Prompt: Combines the retrieved chunks into one piece of text.
retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])

# Initialize the LLM: Loads the Qwen2.5-1.5B-Instruct model for text generation.
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

# Manually pass retrieved text to LLM
prompt = f"Based on the following text, amswer the question: {query}\n \n {retrieved_text}"   # Gives the LLM both the question and the relevant information retrieved from the document.
answer = llm.predict(prompt)    # The LLM generates an answer based on the retrieved document content.

# Print the answer
print("Answer:", answer)