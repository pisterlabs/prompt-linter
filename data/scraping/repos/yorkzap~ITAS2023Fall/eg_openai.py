from openai import OpenAI

client = OpenAI()

generated_json = input("What info do you want?")

completion = client.chat.completions.create(
    model='gpt-4',
    messages=[
        {'role': 'system', 'content': f'You are an API that provides only JSON responses'},
        {'role': 'user', 'content': f'What kind of response do you want {generated_json}'}
    ]
)

print(completion.choices[0].message.content)
