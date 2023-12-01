from flask import jsonify
import openai

# get keys from .env file
import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

model_id = "MODEL_ID_HERE" ############# CHANGE THIS #############

def get_from_openAI(data): 
    try:
        completion = openai.ChatCompletion.create(
            model=model_id,
            temperature=0.0,
            messages=[
                {"role": "system", "content": "You are a helpful and professional customer service representative, If you don't know something, just say, I don't know"},
                {"role": "user", "content": data['prompt']},
            ]
        )

        data['response'] = completion.choices[0].message["content"]
        return jsonify(data)
    
        # print(completion.choices[0].message["content"])
        # sys.exit(0)

    except Exception as e:
        #Handle service unavailable error
        # print(f"OpenAI API request failed: {e}")
        data['response'] = f"Error: {e}"
        return jsonify(data)