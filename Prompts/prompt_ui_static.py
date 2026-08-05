from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import streamlit as st

# Set up the HuggingFace model and pipeline
llm = HuggingFacePipeline.from_model_id(
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", # Model ID for the Hugging Face model
    task = "text-generation", # Task type for the model
    pipeline_kwargs = {   # Additional arguments for the pipeline
        "max_length": 512,  
        "temperature": 0.5  # Sampling temperature for text generation(tells how much randomness to introduce in the output at each step)
    }
) 

model = ChatHuggingFace(llm = llm)  # Initialize the chat model

# Streamlit UI
st.header("Research Tool")
user_input = st.text_input('Enter your prompt')

if st.button('Summarize'):
    result = model.invoke(user_input)       
    st.write(result.content)

