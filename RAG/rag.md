## RAG
RAG stands for Retrieval-Augmented Generation. It's a technique that makes AI models more accurate by allowing them to look up relevant information before generating an answer.

Think of it like an open-book exam:

Without RAG: The AI answers only from what it learned during training.
With RAG: The AI first searches a knowledge base for relevant documents, then uses those documents to answer.

# Components of a RAG system

                 Documents
                     │
                     ▼
               Text Extraction
                     │
                     ▼
                 Chunking
                     │
                     ▼
               Embedding Model
                     │
                     ▼
              Vector Database
                     ▲
                     │
             User Question
                     │
                     ▼
             Query Embedding
                     │
                     ▼
             Similarity Search
                     │
             Top-k Chunks
                     │
                     ▼
         Prompt + Retrieved Chunks
                     │
                     ▼
                   LLM
                     │
                     ▼
                Final Answer

# 1. Document loaders:
In RAG, a Document Loader is the component responsible for reading data from a particular source and converting it into a format that LangChain can process.
It is the first step in the RAG pipeline:

PDF / Website / CSV / Word / Text
              ↓
       Document Loader
              ↓
      LangChain Documents
              ↓
          Chunking
              ↓
         Embeddings
              ↓
       Vector Database
              ↓
          Retriever
              ↓
             LLM

Some common Document loaders:
1. TextLoader:
Simple and commonly used document loader in LangChain that reads plain text (.txt) files and converts them into LangChain Document objects.

Use Case: 
Ideal for loading chat logs, scraped text, transcripts, code snippets, or any plain text data into a Langchain pipeline.

Limitation:
Works only with .txt files 

2. PyPDFLoader
Used to load content from PDF files and convert each page into a Document object.

Limitations:
It uses the PyPDF library under the hood - not great with scannedPDFS or complex layouts.

3. Directory Loader:
Lets you load multiple documents from a directory(folder) or files.
DirectoryLoader is used when you have multiple files inside a folder and want to load them all into your RAG pipeline at once.

* Syntax:
from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader(
    "data",
    glob="*.txt",
    loader_cls=TextLoader
)

docs = loader.load()

Limitations:
DirectoryLoader normally uses one loader class for the matched files. So if your folder contains both .pdf and .txt files, don't simply use one loader_cls expecting it to handle both formats.

For mixed file types, you can either use separate loaders or configure the loading strategy accordingly.

# lazy_load() in LangChain
lazy_load() is a method used by document loaders to load documents one at a time instead of loading all documents into memory at once.
This is especially useful in RAG when you have a large number of files.

* Syntax:
from langchain_community.document_loaders import DirectoryLoader, TextLoader

loader = DirectoryLoader(
    "data",
    glob="*.txt",
    loader_cls=TextLoader
)

for document in loader.lazy_load():
    print(document.page_content)

4. WebBaseLoader:
WebBaseLoader is a document loader used to extract text/content from web pages so that you can use that content in a RAG pipeline.

* Syntax:
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(
    "https://example.com"
)

documents = loader.load()

print(documents)

The loader fetches the webpage and converts its content into LangChain Document objects.

It uses BeautifulSoup under the hood to parse HTML and extract visible text.

Used for blogs, news articles, or public websites where the content is primarily text-based and static.

Limitations:
Doesn't handle javascript-heavy pages well(use SeleniumURLoader for that),
Loads only static content(What's in the HTML, not what loads after the page renders).

# 2. Text Splitting:
Text Splitting is the process of breaking large chunks of text(like articles, PDFs, HTML pages, or books) into smaller, manageable pieces(chunks) that an LLM can handle effectively.

Why text splitting?
a) Overcoming model limitations: Many embedding models and language models have maximum input size constraints. Splitting allows us to process documents that would otherwise exceed these limits.

b) Downstream tasks: Text splitting improves nearly every LLM powered task like embedding, semantic search, summarization.

c) Optimizing computational resources: Working wiith smaller chunks of text can be more memory efficient and allow for better parallelization of processing tasks. 

1. Length based Text Splitting:
Length-based text splitting is a technique that divides continuous text into fixed-size chunks based on character count, word count, or token count, ensuring compliance with strict length limits such as SMS (160 chars), Twitter/X (280 chars), or LLM context windows.

* from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
    separator = ''
)

splitter.split_text(text)  // if text is to be divided into chunks
splitter.split_documents(docs)  // if pdf or files are to be divided into chunks

Chunk overlap: Chunk overlap is a document chunking technique in Retrieval Augmented Generation (RAG) where consecutive text segments share a portion of their content to preserve contextual continuity and mitigate information loss at chunk boundaries.  By ensuring that the tail of one chunk is repeated at the head of the next, it prevents critical concepts, entities, or sentences split across artificial boundaries from becoming unretrievable.

2. Text-Structured Based(One of the most commomly used): Text structure-based text splitting is a document chunking strategy that divides text by respecting its inherent hierarchical organization—such as paragraphs, sentences, and words—rather than using arbitrary character or token counts.  This approach is primarily implemented via the RecursiveCharacterTextSplitter in LangChain, which attempts to keep larger units (like paragraphs) intact and only splits them into sentences or words if they exceed the specified chunk size. 

Key characteristics and related methods include:

Recursive Hierarchy: The splitter uses a defined list of separators (e.g., \n\n for paragraphs, \n for lines, . for sentences, and spaces for words) to recursively break down text while maintaining semantic coherence. 

Document-Specific Splitting: For structured formats like HTML, Markdown, or JSON, splitters can target specific structural elements (such as HTML tags or Markdown headers) to preserve logical grouping and context. 

Advantages: This method produces chunks that are more readable and semantically meaningful, improving retrieval accuracy in RAG applications by preventing the splitting of related content mid-thought. 

* from langchain_text_splitters import RecursiveCharacterTextSplitterCharacterTextSplitter   

text = """   """
splitter = RecursiveCharacterTextSplitterCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 0,
    separator = ''
)

result = splitter.split_text(text)

print(result)

3. Document-Structured Based:  Document structure-based text splitting is a strategy that segments text by leveraging inherent document syntax—such as Markdown headers, HTML tags, or code blocks—rather than arbitrary character counts.  This approach preserves the logical organization and semantic coherence of the content, making it particularly effective for retrieval-augmented generation (RAG) and summarization tasks where maintaining context is critical. 

Key implementations include:

LangChain’s Document-Structure Splitters: These tools split content based on specific formats. For example, MarkdownTextSplitter respects header hierarchies (#, ##), HTMLTextSplitter uses tags, and RecursiveCharacterTextSplitter can be configured to parse code by functions or classes. 

RecursiveCharacterTextSplitter: A versatile tool that attempts to split by structure first (paragraphs, then sentences, then words) to maintain natural language flow, falling back to character-level splits only when necessary to meet size limits. 

Semantic and AI-Based Splitting: Advanced methods use NLP and semantic analysis to identify logical boundaries based on meaning, topic shifts, or entity changes, which is useful for unstructured documents like legal contracts or medical records where explicit structural markers are absent. 

4. Semantic Meaning Based: Semantic meaning based text splitting (also known as semantic chunking) is a document processing technique that segments text based on semantic similarity rather than fixed character counts or arbitrary token limits.  This method uses embedding models to analyze the meaning of sentences or paragraphs, identifying natural breakpoints where the topic or context shifts significantly. 

Key aspects of semantic text splitting include:

Context Preservation: By splitting at points of low semantic similarity, the resulting chunks retain their contextual integrity, which improves the quality of Retrieval-Augmented Generation (RAG) systems.

Embedding-Based Analysis: The process typically involves converting text segments into vector embeddings and calculating cosine distance or similarity scores to detect shifts in meaning. 

Threshold Configuration: Users can adjust parameters like breakpoint thresholds or percentile dissimilarity to control the granularity of the splits, balancing between fewer, larger chunks and more, smaller segments. 

# 3. Vector Stores:
A vector store is a system designed to store and retrieve data represented as numerical vectors. 
Vector stores serve as the critical storage and indexing layer in RAG (Retrieval-Augmented Generation) pipelines, holding vector embeddings that represent the semantic meaning of documents.  Unlike traditional keyword search, these stores use approximate nearest neighbor (ANN) algorithms to retrieve relevant context based on geometric proximity, enabling LLMs to generate factually grounded responses.

In production, vector databases extend basic vector stores by adding durability, metadata filtering, hybrid search (combining semantic and lexical relevance), and access control.  Key open-source options include pgvector, Qdrant, Weaviate, and Milvus, while managed services like Pinecone and DataStax Astra DB offer serverless scalability. 

Selecting the right store depends on scale and requirements: in-memory stores like Chroma suit prototyping, while distributed systems handle enterprise-grade multi-step agent workflows and high-throughput retrieval. Effective RAG systems often pair vector stores with external object storage (e.g., S3) for raw documents to optimize cost and performance. 

Key features:
Storage: Ensures that vectors and their associated metadata are retained, whether in-memory for quick lookups or on-disk for durability and large-scale use.

Similarity search: Helps retrieve the vectors most similat to a query vector.

Indexing: Provide a data structure or method that enables fast similarity searches on high-dimensional vectors.

CRUD Operations: Manage the lifecycle of data- adding new vectors, reading them, updating existing entries, removing outdated vectors.

## Vector Store Vs Vector Database
Vector Store:
A vector store is a specialized type of data management system designed to store and retrieve vector embeddings. Think of it as a lightweight library or feature, often integrated within a larger system, primarily focused on handling numerical representations of data. Vector embeddings are crucial in AI because they convert complex information, like text, images, or audio, into a format that machines can easily understand and compare.
The primary role of a vector store is to provide an efficient mechanism for performing similarity searches. When you have a piece of data, such as a search query, you can convert it into a vector. The vector store then helps you find the most similar vectors among those it’s already stored. This process, known as approximate nearest neighbor (ANN) search, delivers fast, relevant results, even with millions of data points.

Vector Database:
A vector database is a purpose-built database created specifically to store, manage, and query high-dimensional vector embeddings at scale. While a vector store offers foundational capabilities for handling vectors, a vector database is a much more robust, feature-rich system. It’s designed from the ground up to handle the complexities of massive vector datasets, providing the scalability, performance, and reliability required for enterprise-grade applications.
Unlike a simple vector store, which might be a library or an extension within another system, a vector database is a standalone solution. It provides a full suite of database management features, including data persistence, advanced indexing, security controls, and support for complex queries. These capabilities make it a good choice for organizations that need to manage billions or even trillions of vectors while ensuring fast and accurate retrieval.
Key capabilities include:
Hybrid data storage: Supports both vector embeddings and traditional data types (text, numbers, metadata), allowing unified querying across multiple data formats.

Advanced querying: Enables complex queries that mix vector similarity search with filters, aggregations, and Boolean logic–similar to SQL-style operations.

Data persistence and durability: Ensures vectors and metadata are securely stored and recoverable, even after restarts or system failures.

Index management: Automatically handles creation, optimization, and scaling of vector indexes for fast similarity search performance.

Scalability and distribution: Designed for horizontal scaling across clusters, supporting high-throughput workloads and global deployments.

Integration and APIs: Provides REST, gRPC, or SDK-based APIs for seamless integration with AI models, data pipelines, and application frameworks.

Security and access control: Includes authentication, authorization, and encryption features to protect sensitive data in enterprise environments.

Observability and monitoring: Offers tools to track query performance, index health, and resource utilization for optimized system management.

A vector database is effectively a vector store with extra database features(eg, clustering, scaling, security, metadata filtering, durability)

# Chroma Vector Store:
Chroma is a lightweight open source vector database that is especially friendly for local development and small to medium scale production needs. 

# 4. Retiever:
A retiever is a component in Langchain that fetches relevant documents from a data source in response to user's query. 
Unlike vector stores, retrievers do not need to store documents; they only need to retrieve them, allowing vector stores to be converted into retrievers using the .as_retriever() method. 

There are multiple types of retrievers.

All retrievers in LangChain are runnables, making them plug-and-play components that can be easily integrated into larger LangChain chains or RAG (Retrieval-Augmented Generation) workflows.  They bridge the gap between unstructured user queries and structured data, offering more advanced search strategies than basic vector similarity searches. 

Common types of retrievers include:

Vector Store Retrievers: The most common type, searching through vector databases like Chroma or Pinecone using embeddings. 

Multi-Query Retriever: Uses an LLM to generate multiple queries from a single user prompt to improve coverage and accuracy. 

Contextual Compression Retriever: Uses a base retriever to find documents and then applies a compressor (often an LLM) to strip away irrelevant content, returning only the most pertinent text. 

External Index Retrievers: Search sources like Wikipedia, Arxiv, or internet search APIs (e.g., Perplexity, Parallel Search). 

Maximum Marginal Relevance: MMR is an information retrieval algorithm designed to reduce redundancy in the retrieved results while maintaining high relevance to the query. It tells "How can we pick results thst are not only relevant to the query but also different from each other."

## RAG(Retrieval Augmented Generation)
Retrieval augmented generation, or RAG, is an architecture for optimizing the performance of an artificial intelligence (AI) model by connecting it with external knowledge bases. RAG helps large language models (LLMs) deliver more relevant responses at a higher quality.

RAG allows generative AI models to access additional external knowledge bases, such as internal organizational data, scholarly journals and specialized datasets. By integrating relevant information into the generation process, chatbots and other natural language processing (NLP) tools can create more accurate domain-specific content without needing further training.

# What are the benefits of RAG?
The primary benefits of RAG include:

Cost-efficient AI implementation and AI scaling
Access to current domain-specific data
Lower risk of AI hallucinations
Increased user trust
Expanded use cases
Enhanced developer control and model maintenance
Greater data security

# How does RAG work?
RAG systems follow a five-stage process:

1. The user submits a prompt.
 

2. The information retrieval model queries the knowledge base for relevant data.
 

3. Relevant information is returned from the knowledge base to the integration layer.
 

4. The RAG system engineers an augmented prompt to the LLM with enhanced context from the retrieved data.
 

5. The LLM generates an output and returns an output to the user.

# Components of a RAG system
RAG systems contain four primary components:

The knowledge base: The external data repository for the system.

The retriever: An AI model that searches the knowledge base for relevant data.

The integration layer: The portion of the RAG architecture that coordinates its overall functioning.

The generator: A generative AI model that creates an output based on the user query and retrieved data.
