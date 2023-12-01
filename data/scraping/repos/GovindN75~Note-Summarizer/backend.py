import time
import cohere
from flask import Flask, request, jsonify
from flask_cors import CORS
import re


app = Flask(__name__)
CORS(app)
api_key = 'oF6eA5FnAgLKeezfIAgjWn7PraIRJHH00riUjr5Q' 
co = cohere.Client(api_key)
MAX_STRING_SIZE = 1000

# Split the prompt into chunks of 500 characters
def pre_process_prompt(prompt):
    prompt_array = []
    if len(prompt) > MAX_STRING_SIZE:
        while prompt:
            idx = prompt[:MAX_STRING_SIZE].rfind('.')
            
            
            if idx == -1:
                idx = prompt.find('.')
            
            
            if idx == -1:
                idx = len(prompt)

            chunk = prompt[:idx+1]  
            prompt_array.append(chunk)  
            prompt = prompt[idx+1:]  
            
        return prompt_array
    
    return [prompt]

@app.route('/api/summarize', methods=['POST'])
def summarize_text():
    request_json = request.json
    
    prompt = request.json.get('text')
    prompt_array = pre_process_prompt(prompt)
    
    summary = []
    
    format = request_json.get('format').lower()
    summary_length = request_json.get('summary_length').lower()
    for i, input_prompt in enumerate(prompt_array):
        response = co.summarize(
        length=summary_length,
        text=prompt,
        format=format,
        model='summarize-medium',
        additional_command='',
        temperature=0.1,
        )
        if format == "bullets":
            summary += (response.summary.split('\n'))
        else:
            summary.append(response.summary)
        if i != 0:
            time.sleep(15) # rate limiting
            
    return summary if format == "bullets" else [' '.join(summary)]


if __name__ == '__main__':
    app.run(debug=True)

