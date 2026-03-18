from ollama import Client
import sys

NEBULA_04_BASE = "http://ece-nebula04.eng.uwaterloo.ca:11434"

def ollama_generate(prompt: str) -> str:
    """
    Interfaces with the Qwen 80B model on the nebula04 cluster.
    """
    print(f"Connecting to {NEBULA_04_BASE}...")
    nebula04_client = Client(host=NEBULA_04_BASE)
    try:
        response = nebula04_client.chat(
            model='qwen3-next:latest',
            messages=[{'role': 'user', 'content': prompt}],
            options={
                "num_ctx": 125000, # Hard limit
                "temperature": 0.5 # Lower temp for better code generation
            }
        )
        return response['message']['content']
    except Exception as e:
        print(f"[API ERROR] {e}")
        return "" # Return empty on failure to prevent crash

if __name__ == "__main__":
    prompt = "Hello, are you ready for the experiment? Please reply with 'I am ready'."
    print(f"Sending prompt: '{prompt}'")
    response = ollama_generate(prompt)
    print("\n--- Model Response ---")
    print(response)
    print("----------------------")

    if response:
        print("Connection successful!")
    else:
        print("Connection failed.")
        sys.exit(1)
