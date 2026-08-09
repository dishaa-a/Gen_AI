# Imports ChatHuggingFace, HuggingFacePipeline, PromptTemplate, and load_dotenv.
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Loads variables stored in the .env file.
load_dotenv()

# Loads the Qwen2.5-1.5B-Instruct model and configures it for text generation.
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

# Create a Prompt Template: Defines a reusable prompt with a {topic} variable.
prompt = PromptTemplate(
    input_variables = ["topic"],
    template = "Suggest a catchy blog title about {topic}."
)

# Define the input: Gets the blog topic from the user.
topic = input("Enter a topic")

# Format the prompt manually using PromptTemplate: Replaces {topic} with the user's input.
formatted_prompt = prompt.format(topic = topic)

# Call the LLM directly: Sends the formatted prompt to the LLM and generates a title.
blog_title = llm.predict(formatted_prompt)

# Print the output
print("Generated blog title: ", blog_title)