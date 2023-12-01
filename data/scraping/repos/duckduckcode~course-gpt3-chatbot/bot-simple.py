# CS4HS 2022
# Write an open-domain chatbot with Python
# Tanya Gray (tanya.gray@soulmachines.com)
#
# bot-simple.py
# =====

import openai

# use your OpenAI account
openai.api_key = "YOUR_API_KEY"

# greet the user
print("Bot: Hello!")

while True:
    # capture user input
    user_input = input("You: ")

    # create a prompt for GPT-3
    prompt = "August is a kind bot who wants to be friends.\n"

    # leave a blank line before the chat history
    prompt += "\n"

    # add example chat history
    prompt += "Bot: Hello!\n"
    prompt += "You: " + user_input + "\n"

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
