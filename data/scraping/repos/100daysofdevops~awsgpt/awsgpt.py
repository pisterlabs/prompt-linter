import os
import openai
import sys

# Constants
MODEL_NAME = 'gpt-4'
TEMPERATURE = 0.8
MAX_TOKENS = 200

# List of restricted words related to AWS
RESTRICTED_WORDS = ["delete", "destroy", "terminate"]

# Initialize OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def is_restricted_text(text):
    for word in RESTRICTED_WORDS:
        if word in text.lower():
            return True
    return False

def get_output_from_gpt(prompt, operation_type):
    try:
        gpt_response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a Python code generator assistant specialized in AWS. You can also assist in searching for AWS-related information."},
                {"role": "user", "content": prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        
        full_text = gpt_response['choices'][0]['message']['content'].strip()
        
        if is_restricted_text(full_text):
            return "Sorry, I can't execute that operation due to restricted words."
        
        if operation_type == "Generate Code":
            start_idx = full_text.find("```python")
            end_idx = full_text.find("```", start_idx + len("```python"))

            if start_idx == -1 or end_idx == -1:
                return "Could not find a Python code block."
            
            code_block = full_text[start_idx + len("```python"):end_idx].strip()
            return code_block
        elif operation_type == "Generate Search":
            return full_text
    except Exception as e:
        return f"Error interacting with OpenAI: {e}"

if __name__ == '__main__':
    prompt = sys.stdin.readline().strip()  
    operation_type = sys.stdin.readline().strip()  
    generated_output = get_output_from_gpt(prompt, operation_type)
    
    if generated_output:
        print(generated_output)  
