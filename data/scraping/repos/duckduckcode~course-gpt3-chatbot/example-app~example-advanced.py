import openai
import json

# use your OpenAI account
openai.api_key = "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"

# write some basic facts about your bot.
# these will be added to every prompt.
facts = [
    "The bot was created by Tanya Gray.",
    "The bot was created on 24 November 2022.",
    "The bot's favourite colour is blue, like the ocean.",
    "The bot likes to browse the Internet.",
    "The bot is an only child.",
]

message_history = []

# greet the user
print("Bot: Hello!")

# add bot greeting to the message history
message_history.append("Bot: Hello!")

while True:
    # capture user input
    user_input = input("You: ")

    # add user input to the message history
    message_history.append("You: " + user_input)

    # build a prompt for GPT-3
    prompt = "August is a kind bot who wants to be friends.\n"

    # leave a blank line before the chat history
    prompt += "\n"

    # add the facts to the prompt
    for fact in facts:
        prompt += fact
        prompt += "\n"

    # leave a blank line before the chat history
    prompt += "\n"

    # add the chat history to the prompt
    for message in message_history[-6:]:
        prompt += message
        prompt += "\n"

    # add the empty line for the bot to respond
    prompt += "Bot:"

    # optional line to preview your prompt in the command line
    # print("\n===\n" + prompt + "\n===\n")

    # send the GPT3 request
    completion = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        temperature=0,
        max_tokens=150,
        stop="You:"
    )

    # print("\n===\n" + json.dumps(completion, indent=2) + "\n===\n")

    # extract the response
    bot_response = completion.choices[0].text.strip()

    # display the response
    print("Bot: " + bot_response)

    # add bot response to the message history
    message_history.append("Bot: " + bot_response)
