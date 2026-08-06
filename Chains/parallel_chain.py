from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

llm1 = HuggingFacePipeline.from_model_id(
    model_id = "Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 512,
        "temperature": 0.5,
        "do_sample": True,
        "return_full_text": False,   # <-- key fix: only return the new reply
    }
)

llm2 = HuggingFacePipeline.from_model_id(
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 512,
        "temperature": 0.5,
        "do_sample": True,
        "return_full_text": False,   # <-- key fix: only return the new reply
    }
)

# Model
model1 = ChatHuggingFace(llm=llm1)

model2 = ChatHuggingFace(llm=llm2)

prompt1 = PromptTemplate(
    template = 'Generate short and simple notes from the following test \n {text}',
    input_variables = ['text']
)

prompt2 = PromptTemplate(
    template = 'Generate 5 short question answers from the following text \n {text}',
    input_variables = ['text']
)

prompt3 = PromptTemplate(
    template = 'Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables = ['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser 
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """Support Vector Machines (SVM) are supervised machine learning algorithms primarily used for classification and regression tasks.  Developed in the 1990s by Vladimir Vapnik and colleagues, SVMs work by finding an optimal hyperplane in an N-dimensional space that maximizes the margin between the closest data points of different classes, known as support vectors. 

Key characteristics include:

Kernel Trick: SVMs handle non-linearly separable data by using kernel functions (e.g., linear, polynomial, radial basis function) to map data into higher-dimensional spaces for linear separation. 
Types: Linear SVMs are used for linearly separable data, while non-linear SVMs use kernels for complex boundaries.  Support Vector Regression (SVR) extends the concept to predict continuous values.
Applications: Commonly used in natural language processing (sentiment analysis, spam detection), image classification, and bioinformatics due to their effectiveness with high-dimensional data. 
Advantages: Effective in high-dimensional spaces, memory efficient, and resilient to overfitting, though they can be computationally expensive for large datasets."""

result = chain.invoke({'text': text})

print(result)