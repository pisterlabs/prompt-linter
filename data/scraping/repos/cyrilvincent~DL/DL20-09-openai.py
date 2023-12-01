from openai import OpenAI

with open("data/chatgpt/openai.env") as f:
    key = f.read()

client = OpenAI(api_key=key)

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "Tu es un formateur en informatique"},
    {"role": "user", "content": "Expliques moi ce qu'est le language Python ?"}
  ]
)

res = completion.choices[0].message
print(res.content)
