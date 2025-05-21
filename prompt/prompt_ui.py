from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint # Import Hugging Face specific classes
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt
import os # Import os to check for HUGGINGFACEHUB_API_TOKEN

load_dotenv() # Load all environment variables from the .env file

# --- Hugging Face Model Initialization ---
# Check if HUGGINGFACEHUB_API_TOKEN is available
hf_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_api_token:
    st.error("HUGGINGFACEHUB_API_TOKEN not found in environment variables. Please set it in your .env file.")
    st.stop() # Stop the app if the token is missing

try:
    # Initialize the Hugging Face Endpoint model (using API)
    # Using the model that previously worked for you via the API
    llm_hf = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        temperature=0.1, # Keep temperature low for more factual summaries
        max_new_tokens=500, # Increased tokens for potentially longer summaries
        # No 'task' parameter here, as ChatHuggingFace handles the chat formatting
    )
    # Wrap it with ChatHuggingFace for a consistent chat interface
    model = ChatHuggingFace(llm=llm_hf)
    st.success("Hugging Face model (HuggingFaceH4/zephyr-7b-beta) loaded successfully using Inference API.")
except Exception as e:
    st.error(f"Error loading Hugging Face model: {e}")
    st.stop() # Stop the app if the model fails to load


st.header('Research Tool (Hugging Face Model)')

paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_input = st.selectbox(
    "Select Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)

template = load_prompt('template.json')

if st.button('Summarize'):
    with st.spinner('Generating summary...'): # Show a spinner while processing
        try:
            chain = template | model
            result = chain.invoke({
                'paper_input': paper_input,
                'style_input': style_input,
                'length_input': length_input
            })
            st.subheader("Summary:")
            st.write(result.content)
        except Exception as e:
            st.error(f"Error during summarization: {e}. Please try again or check logs.")