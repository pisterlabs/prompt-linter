import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

class Conversation:
    def __init__(self, system, model, max_tokens=17, temperature=0):
        self.prompt = [{"role": "system", "content": system}]
        self.model = model
        self.utilized_tokens = 0
        self.max_tokens = max_tokens
        self.temperature = temperature

    def add_to_prompt(self, role, text):
        self.prompt.append({"role": role, "content": text})

    def send_request(self):
        resp = openai.ChatCompletion.create(
            model=self.model,
            messages=self.prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        self.utilized_tokens = resp["usage"]["total_tokens"]
        self.add_to_prompt("assistant", resp["choices"][0]["message"]["content"])
        return resp["choices"][0]["message"]["content"]
    
    def print_latest_response(self):
        print(self.prompt[-1]['content'])



if __name__ == '__main__':

    #  "text-davinci-003"
    max_tokens = int(input('How many max tokens are you okay with?: '))
    model = "gpt-3.5-turbo" # input('Which model are you using? (default: text-davinci-003)')
    system = input('Who is the system?:')

    print('-'*3, 'Starting Conversation', '-'*3)
    print(f'System: {system}')
    current_conversation = Conversation(
        system=system,model=model, max_tokens=max_tokens, temperature=0
    )
    end_chat = False
    while end_chat or current_conversation.utilized_tokens < current_conversation.max_tokens:
        next_question = input('You:')

        if next_question == 'end':
            end_chat = True
            continue
        current_conversation.add_to_prompt('user', next_question)
        response = current_conversation.send_request()
        print(f'Agent: {response}')