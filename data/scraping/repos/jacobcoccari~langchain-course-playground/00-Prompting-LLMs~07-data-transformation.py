from dotenv import load_dotenv
import openai
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def chat(message, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": message}],
    )
    return response.choices[0].message["content"]


contacts = [
    {
        "Name": "John Doe",
        "Phone Number": "212-555-1234",
        "Email": "john.doe@example.com",
    },
    {
        "Name": "Jane Smith",
        "Phone Number": "212-555-5678",
        "Email": "jane.smith@example.com",
    },
    {
        "Name": "Bob Johnson",
        "Phone Number": "212-555-9012",
        "Email": "bob.johnson@example.com",
    },
]

prompt = f""" Please transform the following data into a HTML table:

``` {contacts}```
"""

response = chat(prompt)
print(response)
