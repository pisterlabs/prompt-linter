import openai
from dotenv import load_dotenv
load_dotenv(".gitignore/secrets.sh")
import os

def get_paint_info(color_prompts):
    attempt_count = 0
    max_attempts = 5

    while attempt_count < max_attempts:
        try:

            # define OpenAI key
            api_key = os.getenv("OPENAI_API_KEY")
            openai.api_key = api_key
            
            responses = []
            
            for prompt in color_prompts:
                messages = [
                    {"role": "system", "content": "You are a customer in a paint store."},
                    {"role": "user", "content": prompt}
                ]
                
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                )
                
                responses.append(response)
            
            return [response['choices'][0]['message']['content'] for response in responses]

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            attempt_count += 1

    print("Max attempts reached. Function failed.")
    return None

def send_message(message):
    url = 'https://api.openai.com/v1/chat/completions'
    headers = {
        'Authorization': 'Bearer YOUR_API_KEY',
        'Content-Type': 'application/json'
    }
    data = {
        'messages': [{'role': 'system', 'content': 'You are a helpful assistant.'},
                     {'role': 'user', 'content': message}]
    }
    response = requests.post(url, headers=headers, json=data)
    response_json = response.json()
    message = response_json['choices'][0]['message']['content']
    return message

