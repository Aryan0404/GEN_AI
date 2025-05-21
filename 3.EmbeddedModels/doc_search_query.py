from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os # Import os to get environment variables

load_dotenv() # Load your HUGGINGFACEHUB_API_TOKEN from .env

# Initialize the embedding model to use the Hugging Face Inference API
# It will pick up HUGGINGFACEHUB_API_TOKEN from the environment.
# As discussed, it defaults to a common embedding model on the HF Inference API.
embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = 'tell me about Rohit Kohli'

# Generate embeddings for the documents using the HF Inference API
doc_embeddings = embedding.embed_documents(documents)

# Generate an embedding for the query using the HF Inference API
# HuggingFaceEndpointEmbeddings directly provides embed_query
query_embedding = embedding.embed_query(query)

# Calculate cosine similarity
scores = cosine_similarity([query_embedding], doc_embeddings)[0]

# Find the index and score of the document with the highest similarity
index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]
print(scores)
print(query)
print(documents[index])
print("similarity score is:", score)