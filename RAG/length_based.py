from langchain_text_splitters import CharacterTextSplitter   
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('SPMM.pdf')

docs = loader.load()

text = """Length-based text splitting is a technique that divides continuous text into fixed-size chunks based on character count, word count, or token count, ensuring compliance with strict length limits such as SMS (160 chars), Twitter/X (280 chars), or LLM context windows.  This method is critical when content integrity must be maintained within technical constraints, though it may occasionally split sentences or words if natural boundaries do not align with the specified size. 

In LangChain, this is implemented through several specific splitters:

CharacterTextSplitter: Splits text by a specified separator and measures chunk size by character count. 
RecursiveCharacterTextSplitter: Iteratively splits text using a hierarchy of separators (paragraphs, sentences, words) until chunks fit within the chunk_size. 
TokenTextSplitter: Divides text based on token counts using HuggingFace tokenizers, which is essential for aligning with how Large Language Models (LLMs) interpret input limits. 
For general-purpose use, online tools allow users to split text by length or delimiter in real-time, while enterprise pipelines like Azure AI Search offer a Text Split skill that can chunk text by pages or sentences with configurable maximum lengths and overlap settings to preserve context across chunks. """

splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
    separator = ''
)

result = splitter.split_documents(docs)

print(result)