import os
import openai

openai.api_key = os.getenv("OPENAI_KEY")
MODEL = "gpt-3.5-turbo-0613"
NUM_RESPONSES = 1
MAX_RESPONSE_TOKENS = 4
TEMPERATURE = 1
FREQUENCY_PENALTY = 1
TOP_P = 1
PRESENCE_PENALTY = 1

def make_chat_completion(prompt):
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_RESPONSE_TOKENS,
        top_p=TOP_P,
        n=NUM_RESPONSES,
        frequency_penalty=FREQUENCY_PENALTY,
        presence_penalty=PRESENCE_PENALTY
    )
    return response

user_prompt = "Hamburger and "
response = make_chat_completion(user_prompt)

# Print only the text reply
for i in range(NUM_RESPONSES):
    print(response['choices'][i]['message']['content'])
