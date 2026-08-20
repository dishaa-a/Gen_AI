from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

"""Document Structure based Text Splitting"""
python_code = """
def add(a, b):
    return a + b
"""
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300,
    chunk_overlap=0
)
chunks = splitter.split_text(python_code)   

print(chunks)