from openai import OpenAI

client = OpenAI(
    api_key=""
)


def get_chat_response(prompt):
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model="ft:gpt-3.5-turbo-0613:braincore::8OMgZumY",
    )
    return chat_completion.choices[0].message.content


if __name__ == "__main__":
    # Get user input from CLI
    user_prompt = input("Enter your prompt: ")

    # Get and print the response
    response = get_chat_response(user_prompt)
    print(response)
