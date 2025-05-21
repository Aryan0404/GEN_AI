from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
import os

load_dotenv() # Load your HUGGINGFACEHUB_API_TOKEN from .env

# Set the API token as an environment variable (this is what HuggingFaceEndpointEmbeddings expects)
# It's good practice to ensure it's set before initializing the class

# Initialize the embedding model to use the Hugging Face Inference API
# HuggingFaceEndpointEmbeddings will automatically pick up HUGGINGFACEHUB_API_TOKEN
# from the environment. It also generally uses a default model for embeddings
# on the Inference API, or you might specify it via `huggingfacehub_api_args` if needed.
# However, the error message indicates it doesn't take 'model_name' directly.
embedding = HuggingFaceEndpointEmbeddings() # No 'model_name' or 'api_key' here

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

# Generate embeddings for the list of documents using the API
vector = embedding.embed_documents(documents)

print(str(vector))