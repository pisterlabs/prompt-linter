import openai
import json

response = openai.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[
        {
            "role": "system",
            "content": "You are a book suggester. Respond with a book title, description and category in JSON format."
        },
        {
            "role": "user",
            "content": "I want to make lots of money"
        }
    ],
    response_format={"type": "json_object"}
)

print(
    json.dumps(
        json.loads(
            response.choices[0].message.content
        ),
        indent=4
    )
)
