from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint # Import Hugging Face specific classes
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage # For message types
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # For structured chat prompts
from dotenv import load_dotenv
import os

load_dotenv() # Load your HUGGINGFACEHUB_API_TOKEN from .env

# --- Hugging Face Model Initialization ---
hf_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_api_token:
    print("FATAL ERROR: HUGGINGFACEHUB_API_TOKEN not found in environment variables.")
    print("Please set it correctly in your .env file and restart your terminal.")
    exit()

try:
    # Initialize the Hugging Face Endpoint model (using API)
    # This is the actual LLM instance
    llm_hf_endpoint = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        temperature=0.7,
        max_new_tokens=500,
    )
    # Wrap it with ChatHuggingFace for consistent chat interface
    model = ChatHuggingFace(llm=llm_hf_endpoint)
    print("Hugging Face model (HuggingFaceH4/zephyr-7b-beta) loaded successfully using Inference API.")
except Exception as e:
    print(f"FATAL ERROR: Could not load Hugging Face model: {e}")
    print("Please ensure your HUGGINGFACEHUB_API_TOKEN is valid and the model is accessible.")
    exit()

# --- Define the Chat Prompt Template ---
# This template will handle injecting the system message and dynamic chat history
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent. Always be polite and concise.'),
    MessagesPlaceholder(variable_name='chat_history'), # This placeholder will take the list of messages
    ('human', '{user_query}') # Renamed from 'query' to 'user_query' for clarity
])

# --- Initialize Dynamic Chat History ---
# This list will store SystemMessage, HumanMessage, and AIMessage objects
# The system message is part of the template, so the history can start empty or with prior context
chat_history = [] # We'll start with an empty history for a fresh conversation

print("\n--- Starting Hugging Face Chat with Structured Prompt (type 'exit' to quit) ---")

while True:
    user_input = input('You: ')

    if user_input.lower() == 'exit':
        break

    # 1. Create the full prompt using the template and current user input
    # The template will insert the chat_history list and the new user_input
    # The 'system' message is handled by the template itself.
    full_prompt = chat_template.invoke({
        'chat_history': chat_history,
        'user_query': user_input
    })

    try:
        # 2. Invoke the model with the formatted prompt
        # The prompt object contains the list of messages (System, History, Human)
        result = model.invoke(full_prompt.messages) # Pass the list of messages from the prompt object

        # 3. Append the *current* human message and the AI's response to the history
        # This keeps the chat_history list updated for the next turn
        chat_history.append(HumanMessage(content=user_input)) # Add human's message
        chat_history.append(AIMessage(content=result.content)) # Add AI's message

        print("AI: ", result.content)

    except Exception as e:
        print(f"Error communicating with the AI: {e}")
        # If an error occurs, you might want to remove the last user input from history
        # if the AI didn't respond to avoid sending a broken history in the next turn
        if chat_history and chat_history[-1].content == user_input: # Basic check
             chat_history.pop()


print("\n--- Full Chat History (from LangChain objects) ---")
# Print the final chat history in a readable format
for message in chat_history:
    if isinstance(message, SystemMessage):
        print(f"System: {message.content}")
    elif isinstance(message, HumanMessage):
        print(f"You: {message.content}")
    elif isinstance(message, AIMessage):
        print(f"AI: {message.content}")
print("\n--- Chat Ended ---")