import requests
import os
from dotenv import load_dotenv

load_dotenv()

# Get the token from the environment variable
HF_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HF_API_TOKEN:
    print("FATAL ERROR: HUGGINGFACEHUB_API_TOKEN not found in .env file.")
    print("Please ensure it's set correctly and your terminal is restarted.")
    exit()

API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
headers = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json"
}
payload = {
    "inputs": "What is the capital of France?",
    "parameters": {
        "max_new_tokens": 50,
        "temperature": 0.1
    }
}

print(f"--- Testing Direct HTTP POST to Hugging Face Inference API ---")
print(f"API URL: {API_URL}")
print(f"Using token (first 5 chars): {HF_API_TOKEN[:5]}*****")

try:
    response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

    print("\nSUCCESS: Raw API call was successful!")
    print("Status Code:", response.status_code)
    print("Response Content:")
    print(response.json()) # Print the full JSON response

except requests.exceptions.HTTPError as errh:
    print(f"\nFAILED: HTTP Error: {errh}")
    print(f"Response Status Code: {errh.response.status_code}")
    print(f"Response Content: {errh.response.text}") # Get the actual error message from HF API
except requests.exceptions.ConnectionError as errc:
    print(f"\nFAILED: Connection Error: {errc}")
    print("This indicates a network or firewall blocking direct access to the Hugging Face API server.")
except requests.exceptions.Timeout as errt:
    print(f"\nFAILED: Timeout Error: {errt}")
    print("The request timed out. This often points to network congestion or a very slow connection/firewall.")
except requests.exceptions.RequestException as err:
    print(f"\nFAILED: An unexpected requests error occurred: {err}")
except Exception as e:
    print(f"\nFAILED: General Python Error: {type(e).__name__}: {e}")

print("\n--- Direct HTTP Test Complete ---")