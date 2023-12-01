from openai import OpenAI, api_key
import json

def extract_keywords(text):
    client = OpenAI(api_key="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",)

    response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    response_format={ "type": "json_object" },
    messages=[
        {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
        {"role": "user", "content": f"What is the main keyword in this text: {text}?"}
        ]
    )

    return json.loads(response.choices[0].message.content)