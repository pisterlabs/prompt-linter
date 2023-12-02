import speech_recognition as sr
import os
import openai
from enum import Enum
from config import API_KEY

class Actions(Enum):
    do_nothing = 0
    turn_off = 1
    chatGPT = 2
    weather = 3
    explain_code = 4
    rewrite_code = 5

class Brain():
    audio_recognizer = sr.Recognizer()
    memory:list[dict[str,str]] = []

    def process_audio(self, audio):
        try:
            text = self.audio_recognizer\
                    .recognize_google(audio,\
                     language = 'en-IN')
        except sr.UnknownValueError:
            text = '' 
        return text.lower()

    def determine_action(self, text:str) -> int:
        text = text.lower()

        # Is this a bot command?
        is_command = False
        for prefix in ['hey', 'hello', 'yo', 'sup']:
            find = f"{prefix} hal"
            if text.startswith(find):
                is_command = True
                text = text[len(find):]

        if not is_command:
            return Actions.do_nothing

        # Determine action
        if self._is_turn_off(text):
            return Actions.turn_off
        elif self._is_weather(text):
            return Actions.weather
        elif self._is_explain_code(text):
            return Actions.explain_code
        elif self._is_rewrite_code(text):
            return Actions.rewrite_code
        else:
            return Actions.chatGPT

    @staticmethod
    def _is_turn_off(text:str)->bool:
        return 'turn off' in text

    @staticmethod
    def _is_weather(text:str)->bool:
        statements = ["what's the weather like", "weather today"]
        return any([s in text for s in statements])

    @staticmethod
    def _is_explain_code(text:str)->bool:
        statements = ["what does this code do", "explain this code"]
        return any([s in text for s in statements])

    @staticmethod
    def _is_rewrite_code(text:str)->bool:
        statements = ["rewrite this code for me"]
        return any([s in text for s in statements])

    def formulate_response(self, text):
        openai.api_key = API_KEY 

        if len(self.memory) == 0:
            system_text = "You are a friendly AI named Hal. Try to answer my questions in less than 20 words."
            self.memory.append({"role":"system", "content":system_text})

        try:
            self.memory.append({"role":"user", "content":text})
            instance = openai.ChatCompletion.create(
                  model="gpt-3.5-turbo",
                  max_tokens = 200,
                  messages=self.memory
            )
            response = instance.choices[0].message.content
            self.memory.append({"role":"assistant", "content":response})
        except Exception as e:
            print(e)
            self.memory = self.memory[0:len(self.memory)-1]
            response = "Sorry, I ran into an issue, plese try me again."

        return response


