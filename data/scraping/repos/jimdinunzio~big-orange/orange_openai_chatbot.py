from openai_chatbot import OpenAiChatbot

class OrangeOpenAiChatbot(OpenAiChatbot):
    def __init__(self):
        prompt_name ="orange_prompt"
        init_prompt = ""
        with open(f'prompts/{prompt_name}.txt', 'r') as f:
            init_prompt = f.read()
        
        with open(f'prompts/{prompt_name}_messages.txt', 'r') as f:
            prompt_messages = f.read()
        
        intro_line = "In chat mode I can answer questions about myself. Say goodbye to end the chat. What can I answer for you today?"
        super().__init__("gpt-3.5-turbo", init_prompt, intro_line, "Human", "Robot", prompt_messages)
        
if __name__ == "__main__":
    chat_bot = OrangeOpenAiChatbot()
    inp = ""
    print(chat_bot.intro_line)
    while inp != "bye":
        print("> ",end='')
        inp = input()
        if inp == ".log":
            print(chat_bot.get_log())
            continue
        answer = chat_bot.get_response(inp)
        chat_bot.add_to_chat_log(answer)
        print(answer)
        