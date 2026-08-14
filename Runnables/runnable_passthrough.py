from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough

# Load the LLM
llm = HuggingFacePipeline.from_model_id(
    model_id = "Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={                # Sets max_new_tokens, temperature, sampling, and return_full_text=False to control the generated response.
        "max_new_tokens": 512,
        "temperature": 0.5,
        "do_sample": True,
        "return_full_text": False,   # <-- key fix: only return the new reply
    }
)

# Model: Wraps the Hugging Face pipeline into a LangChain chat model.
model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template = 'Write a joke about {topic}',
    input_variable = ['topic']
)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template = 'Expalin the following joke {text}',
    input_variables = ['text']
)

joke_gen_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2, model, parser)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({'topic': 'Cricket'})

print(result)