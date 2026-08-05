from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

# Set up the HuggingFace model and pipeline
llm = HuggingFacePipeline.from_model_id(
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task = "text-generation",
    pipeline_kwargs = {
        "max_length": 512,  
        "temperature": 0.5
    }
) 

# Initialize the chat model
model = ChatHuggingFace(llm = llm)

# Streamlit UI
st.header("Research Tool")

# User Inputs
paper_input = st.selectbox("Select the research paper name", ["Select....", "Attention is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis", "AlphaFold: Using AI for scientific discovery", "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis", "DALL·E: Creating Images from Text", "CLIP: Connecting Text and Images", "StyleGAN: A Style-Based Generator Architecture for Generative Adversarial Networks"])

style_input = st.selectbox("Select the style of summary", ["Select....", "Concise Summary", "Detailed Summary", "Bullet Points Summary", "Technical Summary", "Layman's Summary"])

length_input = st.selectbox("Select the length of summary", ["Select....", "Short", "Medium", "Long"])

# Load the prompt template
template = load_prompt("research_summary_template.json")

# Invoke the model with the user inputs when the button is clicked
if st.button('Summarize'):
    chain = template | model
    result = chain.invoke({
        "paper_title": paper_input,
        "summary_style": style_input,
        "summary_length": length_input
    })

    st.write(result.content)

