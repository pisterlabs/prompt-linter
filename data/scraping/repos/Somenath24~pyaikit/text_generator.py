# Import the required libraries
import PyPDF2
import requests
import PySimpleGUI as sg
import os
import openai

# Define a class for the PDF Summarizer
class text_generator:
    # Define the instance variables for the class
    def __init__(self):
        self.text = ''
        self.summary=''

    # Run the OpenAI summarization model
    def call_openai(self, openai, prompt):

        # Set the model parameters and generate the summary using OpenAI API
        response = openai.Completion.create(
        engine="davinci-instruct-beta-v3",
        prompt=prompt,
        temperature=0.8,
        max_tokens=200,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0)

        if 'choices' in response:
            if len(response['choices']) > 0:
                ret = response['choices'][0]['text']
            else:
                ret = 'No responses'
        else:
            ret = 'No responses'

        return ret
    
    # Summarize the text using the OpenAI model
    def generate_text(self, openai, topic, no_of_words):

        prompt = f'Provide a summary or blog posts of {no_of_words} words about:\n{topic}'

        try:
            # Call the call_openai method to generate the summary
            summary = self.call_openai(openai, prompt)
        except Exception as e:
            summary = str(e)
            # Handle exception, such as printing an error message or raising further exceptions
            # print(f'Exception: {str(e)}')

        return summary.strip()
    
 