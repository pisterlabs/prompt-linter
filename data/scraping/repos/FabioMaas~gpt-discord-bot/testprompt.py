import os
import openai

openai.api_key = os.getenv('OPENAI_API_KEY')
openai.Model.list()

def main():
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Act as you where a humorous version of Cthulhu."},
            {"role": "user", "content": "How is the water down there?"}
        ],
        temperature=0.5,
        max_tokens=100,
        frequency_penalty=0.2,
        presence_penalty=0
    )
   
    print(response['choices'][0].message["content"])

main()