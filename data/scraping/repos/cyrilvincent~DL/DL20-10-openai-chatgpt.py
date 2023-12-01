from openai import OpenAI

with open("data/chatgpt/openai.env") as f:
    key = f.read()

with open("data/chatgpt/python.txt") as f:
    text = f.read()

print(text)

client = OpenAI(api_key=key)

completion = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "Fais moi un résumé en 5 parties de ce text"},
        {"role": "user", "content": text}
    ]
)

res = completion.choices[0].message
print(res.content)
