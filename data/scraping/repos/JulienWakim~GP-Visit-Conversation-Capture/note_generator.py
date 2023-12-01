from openai import OpenAI


def generateNotes(fixed_output):

    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key='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',
    )


    messages = []
    for line in fixed_output.strip().split('\n'):
        speaker, content = line.split(':', 1)
        role = "assistant" if "Doctor" in speaker else "user"
        messages.append({"role": role, "content": content.strip()})

    messages.append(
        {"role": "user", "content": "Please summarize the conversation in the form of doctor's notes."}
    )

    print(messages)

    chat_completion = client.chat.completions.create(
        messages=messages,
        model="gpt-3.5-turbo",
    )

    print(chat_completion.choices[0].message.content)


    # summary_response = client.chat.completions.create(
    #   model="gpt-3.5-turbo",
    #   messages=messages
    # )


# print(summary_response.choices[0].message['content'])