import os
import json
import openai


openai.api_key = os.getenv("OPENAI_API_KEY")


def normalize(input: str) -> dict:
    response = openai.ChatCompletion.create(
      model="gpt-4",
      messages=[
        {
          "role": "system",
          "content": """
            You are a helpful assistant. Given a short biography of a person,
            you can generate a JSON object with the following fields: name, age, address. 
            
            Example:
            
            User: My name is John and was born in 1993. I live in 123 Main St.
            Assistant: 
            {
                "name": "John",
                "age": 30,
                "address": "123 Main St"
            }
          """
        },
        {
          "role": "user",
          "content": input
        }
      ],
      temperature=1,
      max_tokens=256,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    return json.loads(response.choices[0].message.content)


if __name__ == '__main__':
    input = "I was born 25 years ago in Kingston Ave, Brooklyn, NY, where I still live. My name is Patrick"
    normalized = normalize(input)
    print(normalized)