# CS4HS 2022
# Write an open-domain chatbot with Python
# Tanya Gray (tanya.gray@soulmachines.com)
#
# bot-advanced.py
# =====

import openai

# use your OpenAI account
openai.api_key = "YOUR_API_KEY"

# write some basic facts about your bot.
# these will be added to every prompt.
facts = [
    "The bot was created by Tanya Gray.",
    "The bot was created on 24 November 2022.",
    "The bot's favourite colour is blue, like the ocean.",
    "The bot likes to browse the Internet.",
    "The bot is an only child.",
]

# store the message history for bot and user
# so we can add the messages to the prompt each time
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

    # create a prompt for GPT-3
    prompt = "August is a kind bot who wants to be friends.\n"

    # leave a blank line before the facts list
    prompt += "\n"

    # add the facts to the prompt
    for fact in facts:
        prompt += fact
        prompt += "\n"

    # leave a blank line before the chat history
    prompt += "\n"

    # add example chat history
    # using only the last 6 messages in the history
    for message in message_history[-6:]:
        prompt += message
        prompt += "\n"

    # add empty line for GPT-3 to complete
    prompt += "Bot:"

    # optional line to preview your prompt in the command line
    print("\n===\n" + prompt + "\n===\n")

    # send the GPT3 request
    completion = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        temperature=0,
        max_tokens=150,
        stop="You:"
    )

    # extract the response
    bot_response = completion.choices[0].text.strip()

    # display the response
    print("Bot: " + bot_response)

    # add bot response to the message history
    message_history.append("Bot: " + bot_response)
