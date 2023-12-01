import openai
import uuid
import os
import sqlite3
import json
import sys

openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    raise Exception("OPENAI_API_KEY is not set.")

def chat(prompt, model="gpt-3.5-turbo"):
    completion = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    return completion.choices[0].message.content

def generate(username: str) -> str:
    return chat(f"""Generate a profile from the following username: {username}.
A profile has a display_name which is a users given name, for example "John Smith" would be a good display_name for a user with a username of johnsmith23.
A profile has a bio which is 1 to 3 sentences describing what the user does and what they are interested in.
A profile may or may not have an avatar_url which is a link to an image of the user.

The profile must be valid JSON with the following format:
{{
    "display_name": "John Smith",
    "bio": "I am a software engineer at OpenAI.",
    "avatar_url": "https://avatars.githubusercontent.com/u/5192759?v=4"
}}

Return only the JSON object, not the text above.
""")

def parse(body: str) -> dict:
    return json.loads(body)

if __name__ == "__main__":
    user_id = sys.argv[1]
    username = sys.argv[2]
    github_username = sys.argv[3]

    profile = parse(generate(username))

    id = uuid.uuid4()

    print(f"    ('{id}', '{user_id}', '{profile['display_name']}', '{profile['bio']}', '{profile['avatar_url']}', '{github_username}'),")

