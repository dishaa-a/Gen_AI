from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate


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

# Create a Prompt Template
prompt = PromptTemplate(
    input_variables = ["topic"],  # Defines what inpur is needed
    template = "Suggest a catchy blog title about {topic}."
)

# Create an LLMChain 
chain = prompt | model

# Run the chain With a specific topic 
topic = input("Enter a topic")
output = chain.run(topic)

print("Generated Blog Title:", output)