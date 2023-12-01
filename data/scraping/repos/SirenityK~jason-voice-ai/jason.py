import face
import speech
import openai
import json
import time
import pyttsx3

face_recognition_engine = face.Recognition()
mic = speech.Speech

print('Initializing...')
face_recognition_engine.initialize_encodings()
print('Starting...')

class Jason:
    def __init__(self) -> None:
        openai.api_key = 'Insert key here'
        self.engine = pyttsx3.init()
        # self.engine.setProperty('voice', 'HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_ES-MX_SABINA_11.0')
        self.prompt = self.get_prompt()
        self.engine.setProperty('rate', 200)
        
    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()
    
    def get_prompt(self) -> tuple:
        with open('prompt.json', encoding='utf8') as file:
            prompt = json.load(file)
            
        return prompt

    def save_jason_memory(self):
        self.prompt.extend([
            {
                "role": "assistant",
                "content": self.answer
            },
            
            {
                "role": "user",
                "content": '!USER es la persona hablando actualmente.\n\n`!MESSAGE`'
            }
        ])

    def chat(self):
        rec_face = face_recognition_engine.detect_face()
        user_message = mic.recognize()
        print(user_message)
        if not user_message:
            return
        
        self.prompt[-1]['content'] = self.prompt[-1]['content'].replace('!USER', rec_face).replace('!MESSAGE', user_message)
        
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=self.prompt
        )
        
        self.answer = response['choices'][0]['message']['content']
        self.save_jason_memory()
        
        print(self.answer)
        self.speak(self.answer)
        
        # for event in response:
        #     event_text = event['choices'][0]['delta']
        #     self.answer = event_text.get('content', '')
            
        #     print(self.answer, end='', flush=True)
        #     time.sleep(.01)
            
        return

if __name__ == '__main__':
    jason = Jason()
    while True:
        jason.chat()
        print()
        time.sleep(1)
