from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('SPMM.pdf')

docs = loader.load()

print(docs)

print(len(docs))

print(docs[0].page_content)