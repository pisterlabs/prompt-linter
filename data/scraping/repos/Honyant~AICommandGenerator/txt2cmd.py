import sys
import openai
openai.api_key = "INSERT_KEY"


MODEL = "gpt-3.5-turbo"
response = openai.ChatCompletion.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "Give bash commands on mac for given natural language prompts. Do not output ANYTHING other than the pure bash command requested!"},
        {"role": "user", "content": f"{sys.argv[1:]}"}
    ],
    temperature=0)
print(response['choices'][0]['message']['content'])
