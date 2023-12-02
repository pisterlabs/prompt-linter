# !pip install openai

import openai

my_message = """
Write a one liner why HashiCorp's Nomad is the best choice for Machine Learning,

like a funny but over the top commercial"""

completion = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {
            "role": "user",
            "content": my_message,
        },
    ],
)

reply = completion.choices[0].message["content"]

print(reply)
