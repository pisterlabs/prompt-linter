import requests

def ask_openai(question):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer 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",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": question
            }
        ]
    }
    response = requests.post(url, headers=headers, json=data)
    
    if "choices" in response.json() and len(response.json()["choices"]) > 0:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        print(f"Error response from OpenAI: {response.json()}")
        return None

response = ask_openai("Teach me about patience.")
if response:
    print(response)
