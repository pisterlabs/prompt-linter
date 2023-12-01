import openai
from dotenv import load_dotenv
import os
from tkinter import Label
from Functions_keyboard import Functions_keyboard as fk

class Word_predictor:
    def __init__(self, root, text_field):
        load_dotenv('.env')
        self.prompt = 'give me the most probability next words in Portuguese based on the prompt: :'
        self.root = root
        self.text_field = text_field
        self.predictions = []
        self.x = 400

    def request(self):
        text =  fk(text_field=self.text_field).get_text()
        openai.api_key = os.getenv("GPT")
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=self.prompt + text,
            max_tokens=100,
            temperature=0
        )
        print(response)  
        print(type(self.text_field))  
        print("prompt " + self.prompt + text)
        print("text " + text)
        return response.choices[0].text

    def get_prediction(self, prediction):
        prediction = prediction.strip().split()
        print(prediction)
        return prediction

    def show_prediction(self, prediction):
        for i in range(len(prediction)):
            self.predictions.append(Label(self.root, text=prediction[i], height=2, width=5, background="white"))
            self.predictions[i].place(x=self.x, y=0)
            self.predictions[i].bind("<Button-1>", lambda event, text=prediction[i]: self.select_prediction(text))
            self.x += 90
        self.x = 400

    def select_prediction(self, text):
#        text = self.replace_word(text)
        fk(text_field=self.text_field).print_value(text)

    def clear_predictions(self):
        for label in self.predictions:
            label.destroy()
        self.predictions = []
        self.x = 400
        
    def replace_word(self, new_word):
        words = fk(text_field=self.text_field).get_text()
        last_index = -1

        for index, word in enumerate(words):
            if word in new_word:
                last_index = index

        # If the word is found, replace it by creating a new list
        if last_index != -1:
            words[last_index] = new_word
            new_text = ' '.join(words)
            return new_text

    def control_prediction(self):
        print("ok")
        try:
            self.predictions.clear()
        finally:
            prediction = self.request()
            prediction = self.get_prediction(prediction)
            self.show_prediction(prediction)
