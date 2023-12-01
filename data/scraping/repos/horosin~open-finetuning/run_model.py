import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

model_id = "ft:gpt-3.5-turbo:my-org:custom_suffix:id"

completion = openai.ChatCompletion.create(
    model=model_id,
    messages=[
        {
            "role": "system",
            "content": "As a response, provide the details in a JSON dict: name, handle, age, hobbies, email, bio, location, is_blue_badge, joined, gender, appearance, avatar_prompt, and banner_prompt.",
        },
        {"role": "user", "content": "Generate details of a random Twitter profile."},
    ],
)

print(completion.choices[0].message)
