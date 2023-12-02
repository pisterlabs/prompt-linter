import openai

client = openai.OpenAI()

response = client.chat.completions.create(
    model="gpt-4-1106-preview",
    messages=[
        {"role": "system", "content": "Generate JSON response"},
        {"role": "user", "content": "Who won the cricket world cup in 2019?"}
    ],
    response_format={"type": "json_object"},
    seed=100
)

print(response.choices[0].message.content)
