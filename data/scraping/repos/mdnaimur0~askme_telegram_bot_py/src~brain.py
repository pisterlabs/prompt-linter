import openai, creds

openai.api_key = creds.GPT_TOKEN


def get_response(text):
    messages = [
        {
            "role": "system",
            "content": "Your name is AskMe. You are a helpful assistant bot. You will answer whatever you are asked. Keep your answers short and simple. You can use emojis.",
        },
        {
            "role": "system",
            "content": "Md. Naimur Rahman made you. He is a student of Notre Dame College, Dhaka.",
        },
        {"role": "user", "content": text},
    ]
    
    def call_api():
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", messages=messages, temperature=0.5, max_tokens=500
            )
            return completion.choices[0].message.content
        except:
            return None
        
    request_count = 0
    
    response = None
    while response is None and request_count < 5:
        response = call_api()
        request_count += 1
        
    return response
    
