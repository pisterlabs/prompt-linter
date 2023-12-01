import json
import os
import datetime
from abc import ABC, abstractmethod
from typing import Any
from gtts import gTTS
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image
from reportlab.lib.styles import getSampleStyleSheet
from langchain.chains import LLMChain, ConversationChain
import openai
from langchain.chat_models import ChatOpenAI
from chatbot_settings import ChatBotSettings
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.conversation.memory import (ConversationBufferMemory,
                                                  ConversationSummaryMemory,
                                                  ConversationBufferWindowMemory,
                                                  ConversationKGMemory)
from pandasai import PandasAI

import pandas as pd


chatbotSettings = ChatBotSettings()
with open('patient_responses.json', 'r') as f:
    all_responses = json.load(f)


# Open the JSON file
with open('patient_questions.json') as file:
    data = json.load(file)

# Get the list of questions
questions = data['questions']

# Flatten the dictionary inside the 'responses' key and match questions to knowledge base
flattened_data = []
for item in all_responses:
    flattened_dict = {}
    flattened_dict['name'] = item['name']
    flattened_dict['date'] = item['date']
    responses = item['responses']
    for question, response in responses.items():
        
        print(question)
        if question in questions:
            # If a match is found, add the match as a new entry in the dictionary
            flattened_dict[question] = response
    flattened_data.append(flattened_dict)


print("Hello Chiropractor")
# Create DataFrame
df = pd.DataFrame(flattened_data)
df.to_csv('patient_responses.csv', index=False)
print(df)

from pandasai.llm.openai import OpenAI
llm = OpenAI()

# Create an empty list to store questions and responses
questions_and_responses = []

while True:
    pandas_ai = PandasAI(llm, conversational=False)
    user_input = input("What data do you want from your patients? ")
    response = pandas_ai.run(df, prompt=user_input)
    print(response)

    # Store the question and response in a dictionary
    question_response = {
        "question": user_input,
        "response": response
    }

    # Append the question and response to the list
    questions_and_responses.append(question_response)

    # Check if the user wants to continue
    cont = input("Do you want to ask another question? (yes/no): ")
    if cont.lower() != "yes":
        break

# Save the questions and responses to a JSON file
with open('questions_and_responses.json', 'w') as f:
    json.dump(questions_and_responses, f)
