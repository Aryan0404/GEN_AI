from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
import os

load_dotenv()

# Initialize the Hugging Face Endpoint
# This is the underlying LLM communication layer
llm_endpoint = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    temperature=0.1, # Lower temperature helps with more deterministic classification
    top_k=50,
    top_p=0.95,
)

# Pass the endpoint to ChatHuggingFace for a chat-like interface
model = ChatHuggingFace(llm=llm_endpoint)

parser = StrOutputParser()

# Pydantic model for structured sentiment output
class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

# Prompt for sentiment classification
prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into positive or negative. Provide only the sentiment label. \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)

# Chain to classify sentiment
classifier_chain = prompt1 | model | parser2

# Prompt for positive feedback response
prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

# Prompt for negative feedback response
prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

# Combine the original input and the sentiment classification result
# Using explicit dictionary key:value syntax
combined = RunnableParallel({
    'sentiment_result': classifier_chain, # Result of sentiment classification (Feedback Pydantic object)
    'feedback': RunnablePassthrough()     # The original input dictionary {'feedback': '...'}
})

# Define the branching logic based on the sentiment
branch_chain = RunnableBranch(
    # If sentiment is positive, run prompt2 and get response
    (lambda x: x['sentiment_result'].sentiment == 'positive', prompt2 | model | parser),
    # If sentiment is negative, run prompt3 and get response
    (lambda x: x['sentiment_result'].sentiment == 'negative', prompt3 | model | parser),
    # Default branch if sentiment is neither (e.g., if classification fails or is neutral)
    RunnableLambda(lambda x: "Could not determine sentiment or generate response.")
)

# Construct the full chain: combine parallel outputs, then branch
chain = combined | branch_chain

# --- Test Cases ---

# Test with positive feedback
positive_feedback = 'This is a beautiful phone, I love it! The camera is amazing and the battery lasts all day.'
print(f"--- Positive Feedback Test ---")
print(f"Input: '{positive_feedback}'")
response_positive = chain.invoke({'feedback': positive_feedback})
print(f"Response: {response_positive}")

print("\n" + "="*80 + "\n")

# Test with negative feedback
negative_feedback = 'The product arrived broken and I am very disappointed. The quality is terrible.'
print(f"--- Negative Feedback Test ---")
print(f"Input: '{negative_feedback}'")
response_negative = chain.invoke({'feedback': negative_feedback})
print(f"Response: {response_negative}")

print("\n" + "="*80 + "\n")

# Print the ASCII graph of the chain for visualization
print("\nGraph for the chain:")
chain.get_graph().print_ascii()