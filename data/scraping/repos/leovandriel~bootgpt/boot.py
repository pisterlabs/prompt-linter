from openai import OpenAI

exec(
    OpenAI(api_key=open("key.txt").read())
    .chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0.001,
        messages=[
            {"role": "user", "content": open("boot.txt").read()},
        ],
    )
    .choices[0]
    .message.content.replace("```", "#")
)
