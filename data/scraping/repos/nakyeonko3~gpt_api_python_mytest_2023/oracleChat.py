import openai

openai.api_key = "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"
model_id = "gpt-3.5-turbo"


def ChatGPT_conversation(conversation):
    response = openai.ChatCompletion.create(model=model_id, messages=conversation)
    # api_usage = response['usage']
    # print('Total token consumed: {0}'.format(api_usage['total_tokens']))
    # stop means complete
    # print(response['choices'][0].finish_reason)
    # print(response['choices'][0].index)
    conversation.append(
        {
            "role": response.choices[0].message.role,
            "content": response.choices[0].message.content,
        }
    )
    return conversation


conversation = []
# conversation.append({"role": "system", "content": "How may I help you?"})
# conversation = ChatGPT_conversation(conversation)
# print(
#     "{0}: {1}\n".format(
#         conversation[-1]["role"].strip(), conversation[-1]["content"].strip()
#     )
# )

while True:
    prompt = input("User:")
    conversation.append(
        {
            "role": "user",
            "content": "In Oracle XE 11G, " + f"{prompt}, " + "answer in English",
        }
    )
    conversation = ChatGPT_conversation(conversation)
    print(
        "{0}: {1}\n".format(
            conversation[-1]["role"].strip(), conversation[-1]["content"].strip()
        )
    )
