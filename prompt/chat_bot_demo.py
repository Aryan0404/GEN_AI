from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint # Import Hugging Face specific classes
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os # Import os to check for HUGGINGFACEHUB_API_TOKEN

load_dotenv() # Load your HUGGINGFACEHUB_API_TOKEN from .env

# --- Hugging Face Model Initialization ---
hf_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_api_token:
    print("FATAL ERROR: HUGGINGFACEHUB_API_TOKEN not found in environment variables.")
    print("Please set it correctly in your .env file and restart your terminal.")
    exit() # Exit the script if the token is missing

try:
    # Initialize the Hugging Face Endpoint model (using API)
    model = ChatHuggingFace(
        llm=HuggingFaceEndpoint(
            repo_id="HuggingFaceH4/zephyr-7b-beta", # Model that previously worked for you
            temperature=0.7, # Adjust temperature as desired
            max_new_tokens=500, # Max tokens for AI's response
            # No 'task' parameter needed here; ChatHuggingFace handles chat formatting
        )
    )
    print("Hugging Face model (HuggingFaceH4/zephyr-7b-beta) loaded successfully using Inference API.")
except Exception as e:
    print(f"FATAL ERROR: Could not load Hugging Face model: {e}")
    print("Please ensure your HUGGINGFACEHUB_API_TOKEN is valid and the model is accessible.")
    exit() # Exit the script if the model fails to load


chat_history = [
    SystemMessage(content='You are a helpful AI assistant')
]

print("\n--- Starting Hugging Face Chat (type 'exit' to quit) ---")

while True:
    user_input = input('You: ')
    
    if user_input.lower() == 'exit': # Use .lower() for case-insensitive exit
        break
        
    chat_history.append(HumanMessage(content=user_input))
    
    try:
        # Invoke the model with the entire chat history
        # Hugging Face models can sometimes be slower, so be patient
        result = model.invoke(chat_history)
        chat_history.append(AIMessage(content=result.content))
        print("AI: ", result.content)
    except Exception as e:
        print(f"Error communicating with the AI: {e}")
        # Optionally, you might want to remove the last HumanMessage if the AI didn't respond
        # chat_history.pop() # Uncomment if you want to remove the last user message on error

print("\n--- Chat History ---")
for message in chat_history:
    if isinstance(message, SystemMessage):
        print(f"System: {message.content}")
    elif isinstance(message, HumanMessage):
        print(f"You: {message.content}")
    elif isinstance(message, AIMessage):
        print(f"AI: {message.content}")
print("\n--- Chat Ended ---")