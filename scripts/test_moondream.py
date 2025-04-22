from ollama import Client
import time

# Test moondream directly
def test_moondream():
    print("Starting Moondream test...")
    client = Client(host='http://localhost:11434')
    
    # Simple prompt
    prompt = "Describe a scene with a person in the top left and a chair in the bottom right."
    
    try:
        print(f"Sending test prompt to Moondream: {prompt}")
        
        # Add delay to ensure server connection
        time.sleep(1)
        
        response = client.chat(
            model='moondream',
            messages=[{
                'role': 'user',
                'content': prompt
            }],
            options={'temperature': 0.5}
        )
        
        print(f"Response type: {type(response)}")
        print(f"Full response: {response}")
        
        if 'message' in response and 'content' in response['message']:
            description = response['message']['content'].strip()
            print(f"\nGenerated Description: {description}")
        else:
            print("\nUnexpected response format. Missing 'message' or 'content'.")
            print(f"Available keys: {response.keys()}")
    
    except Exception as e:
        print(f"--- ERROR testing Moondream ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {str(e)}")

if __name__ == "__main__":
    test_moondream() 