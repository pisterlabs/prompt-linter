import openai
import json
import os
import time

class ChatBot:
    def __init__(self, role):
        self.current_absolute_path = os.path.abspath('.')
        self._load_prompt_dict()
        self.history = [
            {
                "role": "system", 
                "content": self.prompt_dict[role]
            }
        ]

    def listen_and_speak(self, content):
        self.history.append(            
            {
                "role": "user", 
                "content": content
            }
        )
        while True:
            try:
                reply = openai.ChatCompletion.create(
                    model='gpt-3.5-turbo',
                    # model='gpt-4',
                    messages=self.history,
                )
                return reply['choices'][0]['message'].get('content', '')
            except openai.error.RateLimitError as e:
                print('Reached open ai api limit. sleep for 60 seconds')
                time.sleep(60)
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                break


    def _load_prompt_dict(self):
        with open(os.path.join(self.current_absolute_path, 'prompts.json'), 'r') as json_file:
            self.prompt_dict = json.load(json_file)