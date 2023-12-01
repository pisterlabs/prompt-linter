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


json_schema = """{
    "first_name": "sring",
    "last_name": "string",
}"""


prompt = f"""Answer the user query. 

The output should be formatted as a JSON instance using the following schema:
```{json_schema}```

User query:
<Who discovered the theory of relativity?>
"""
response = chat(prompt)
print(response)
