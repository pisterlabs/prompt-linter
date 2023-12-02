import openai

prompt_extension = """""gpt_chat""" ""


def gpt_chat(dictionary, settings):
    # load api key
    openai.api_key = settings["openai_key"]

    chat = [
        {
            "role": "system",
            "content": "You are a helpful assistant who was created by DarkIndustries, you cannot previde internet funtions, tell the user to go back to the regular interface if they ask for anything internet related, always answer nicely no matter what.",
        }
    ]
    print("Running gpt chat... (type 'quit' to exit)")
    prompt = ""

    while prompt != "quit":
        prompt = input("You: ")
        # add the prompt to the chat
        chat.append({"role": "user", "content": prompt})

        result = openai.ChatCompletion.create(
            messages=chat,
            model="gpt-3.5-turbo",
            max_tokens=1000,
        )

        # add the response to the chat
        chat.append(
            {"role": "assistant", "content": result["choices"][0]["message"]["content"]}
        )
        print(f"""ChatGPT: {result["choices"][0]["message"]["content"]}""")
