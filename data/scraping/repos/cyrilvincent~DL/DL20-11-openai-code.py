from openai import OpenAI

with open("data/chatgpt/openai.env") as f:
    key = f.read()

with open("DL08-06-mlp.py") as f:
    text = f.read()

print(text)

client = OpenAI(api_key=key)

completion = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "Voici un code Python, merci de me l'expliquer de manière concise"},
        {"role": "user", "content": text}
    ]
)

res = completion.choices[0].message
print(res.content)
