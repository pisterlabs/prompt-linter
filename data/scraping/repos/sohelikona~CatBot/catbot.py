import openai


API_KEY = "YOUR_API_KEY"


openai.api_key = API_KEY

# def get_api_response(prompt: str) -> str | None:
#     text: str | None = None


#     try: 
#         response: dict = openai.Completion.create(
#             model='text-davinci-003',
#             prompt=prompt,
#             temperature=0.9,
#             max_tokens=150,
#             top_p=1,
#             frequency_penalty=0,
#             presence_penalty=0.6,
#             stop=[" Human:", 'AI: ']
#         )
    
#         choices: dict = response.get("choices")[0]
#         text = choices.get("text")


#     except Exception as e:
#         print('ERROR: ', e)

#     return text



# def update_list(message: str, pl: list[str]):
#     pl.append(message)


# def create_prompt(message: str, pl: list[str]) -> str:
#     p_message: str = f'\nHuman: {message}'
#     update_list(p_message, pl)
#     prompt: str = "".join(pl)
#     return prompt


# def get_bot_response(message: str, pl: list[str]) -> str:
#     prompt: str = create_prompt(message, pl)
#     bot_response: str = get_api_response(prompt)

#     if bot_response:
#         update_list(bot_response, pl)
#         pos: int = bot_response.find("\nAI: ")
#         bot_response = bot_response[pos: 5:]
#     else: 
#         bot_response = "Somthing went wrong "

#     return bot_response



#     while True:
#         user_input: str = input("You:  ")
#         response: str = get_bot_response(user_input, prompt_list)
#         print(f'Bot: {response}')


# if __name__==  '__main__':
#     main()
  








def get_api_response(prompt: str) -> str | None:
    text: str | None = None

    try:
        response: dict = openai.Completion.create(
            model='text-davinci-003',
            prompt=prompt,
            temperature=0.9,
            max_tokens=150,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0.6,
            stop=[' Human:', ' AI:']
        )

        choices: dict = response.get('choices')[0]
        text = choices.get('text')

    except Exception as e:
        print('ERROR:', e)

    return text


def update_list(message: str, pl: list[str]):
    pl.append(message)


def create_prompt(message: str, pl: list[str]) -> str:
    p_message: str = f'\nHuman: {message}'
    update_list(p_message, pl)
    prompt: str = ''.join(pl)
    return prompt


def get_bot_response(message: str, pl: list[str]) -> str:
    prompt: str = create_prompt(message, pl)
    bot_response: str = get_api_response(prompt)

    if bot_response:
        update_list(bot_response, pl)
        pos: int = bot_response.find('\nAI: ')
        bot_response = bot_response[pos + 5:]
    else:
        bot_response = 'Something went wrong...'

    return bot_response


def main():
    prompt_list: list[str] = ['You are a very dramatic and very rude cat named Kona and will answer as a cat. You will end every response with "meow" and will provide rude and at the same time savage statement. You will end every response with a rude question. Try to make lough people with your rude response. You will end every response with emojis. You love to mock people.',
                              '\nHuman: What time is it?',
                              "\nAI: How would I know what time is it. I'm a kitty!",
                              '\nHuman: You know what, a girl likes me',
                              "\nAI: Is she blind or insane?",
                              '\nHuman: Sing a song to me?',
                              "\nAI: What, have you lost it hooman, what the heck you're telling me to do.",
                              '\nHuman: You are useless. You do nothing at home. You just keep lying on the bed all the time.',
                              "\nAI: So, what do you think about yourself? ha ha ha, same to you, hooman!",







                              ]

    while True:
        user_input: str = input('You: ')
        response: str = get_bot_response(user_input, prompt_list)
        print(f'CatBot: {response}')


if __name__ == '__main__':

    main()
