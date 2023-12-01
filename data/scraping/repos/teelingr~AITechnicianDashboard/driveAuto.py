import openai
import os
import json
import requests
from datetime import datetime

openai.api_key = os.getenv("OPENAI_API_KEY")

# The decision tree, intelligent prompting to the technicians while on a live call
class decisionTree():
    """ We have 4 main functions:
        1. Live summary of call
        2. Live fault code detection and resolution
        3. State manager for the panel, controls the state of the frontend panel """

    # 1. Live summary of call
    def liveSummaryModel(chat, existing, existingSummary):
        # print("liveSummary called")
        # print("existing summary was: ", existingSummary)

        # Check if a summary already exists
        if existingSummary:
            # Create a prompt for the GPT-3 model to generate a continuation of the live summary of the call
            prompt = f"The following is a chat log between an SEW-EURODRIVE service technician and one of their customers, they are talking via the customer call center hotline: {chat} Here is an existing summary as to what has happened so far in the call: {existingSummary} If something highly significant has occurred in the chat, that was not mentioned by the existing summary, please respond with the existing summary and another sentence (note the call is still ongoing so do not try to conclude the summary). Do not use more than 8 words per sentence, use a maximum of 50 words for the summary. Here is an example: You either return: 'Serial number provided, customer from Volkswagen, fault code F08.' or 'Serial number provided, customer from Volkswagen, fault code F08. Issue diagnosed to be a faulty motor.'"
        else:
            # Create a prompt for the GPT-3 model to generate a live summary of the call
            prompt = "The following is a chat log between an SEW-EURODRIVE service technician and one of their customers, they are talking via the customer call center hotline. Please create as concise a summary as possible of what has happened so far (note the call is still ongoing so do not try to conclude the summary). Do not use more than 8 words per sentence, use a maximum of 50 words for the summary. Here is an example: 'Serial number provided, customer from Volkswagen, fault code F08. Issue not diagnosed. Issue not solved.'"

        # Send a prompt, with the chat history to the ChatGPT-3.5 model to generate a live summary of the call
        messages = [
            {"role": "user", "content": prompt},
            {"role": "user", "content": str(chat)}
        ]

        try:
            liveSummary = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages = messages) # this is the call to the GPT-3 model

            liveSummary = liveSummary['choices'][0]['message']['content']
            # print("ChatGPT gives a summary of: ", liveSummary[:30], "...") 


            # Check if the live summary is the same as the existing summary, state if info has been added
            if liveSummary == existingSummary:
                infoAdded = False
            else:
                infoAdded = True
                with open('backend/data/liveSummary.json', 'w') as outfile:
                    json.dump({"liveSummary": [liveSummary]}, outfile)

            return liveSummary, infoAdded

        except openai.error.RateLimitError as e:
            print("OpenAI rate limit error")
            liveSummary="Compiling summary..."
            infoAdded=False
            return liveSummary, infoAdded
    

    # 2. Live fault code detection and resolution
    def heardfaultCodeModel(chat, faultCodes):
        # print("heardFaultCodeModel called")

        # Based on the chat, we need to infer the fault code, we do so by creating a prompt for GPT-3
        prompt = f"The following is a chat log between an SEW-EURODRIVE service technician and one of their customers, they are talking via the customer call center hotline: {chat} Please respond with the fault code that was mentioned in the chat. A list of all possible fault codes is: {faultCodes}. Only respond with the fault code, do not add any other information. Here is an example: 'F08'"
        
        # Send a prompt, with the chat history to the ChatGPT-3.5 model
        messages = [
            {"role": "user", "content": prompt},
            {"role": "user", "content": str(chat)}
        ]

        try:
            # this is the call to the GPT-3.5 model
            openai_response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages = messages)

            faultCode = openai_response['choices'][0]['message']['content']
            # print("ChatGPT says fault code: ", faultCode[:3], "...")


            # check if the fault code exists in the fault code list
            if faultCode in faultCodes:
                # print("fault code found in fault code list")
                with open('backend/data/fault_code_database.json', 'r') as outfile:
                    data = json.load(outfile)

                # Extract the possible reason and activity from the fault code database
                possible_reasons = data[faultCode]["possible_reasons"]
                # print("possible reasons: ", possible_reasons)
                activities = data[faultCode]["activities"]
                # print("activities: ", activities)
            else:
                # If the fault code does not exist in the fault code list, we return empty strings
                # print("fault code not found in fault code list")
                possible_reasons = ""
                activities = ""

            return faultCode, possible_reasons, activities

        except openai.error.RateLimitError as e:
            print("OpenAI rate limit error")
            faultCode="Compiling fault code package..."
            possible_reasons= ""
            activities= ""
            return faultCode, possible_reasons, activities


    # 3. This is the main function that calls the other functions.
    def decisionTree(): 
        # print("decisionTree called with chat: \n", chat)

        """The model takes the chat history as an input"""
        # In development use a sample chat history
        # with open("backend/archived/sample_chat_history.json", "r") as file:
        #     data = json.load(file)
        #     chat = data.get("chat", [])

        # When demonstrating use the live chat history
        with open("backend/data/chat_history.json", "r") as file:
            data = json.load(file)
            chat = data.get("chat", [])

        # 1. Live summary of call
        # Check if an existing summary exists
        with open("backend/data/liveSummary.json", "r") as file:
            data = json.load(file)

        # Extract the existing summary, may not exist
        existingSummary = data.get("liveSummary", [])

        if existingSummary:
            # If an existing summary exists, we pass it to the liveSummary function
            existing=True
        else:
            # If an existing summary does not exist, we pass an empty string to the liveSummary function, will create a new one
            existing=False
            existingSummary = ""

        # if chat found
        if chat:

            # 1. Send the chat history to the GPT-3 model to generate a live summary of the call
            liveSummary, infoAdded = decisionTree.liveSummaryModel(chat, existing, existingSummary)
            # print("liveSummary: ", liveSummary)

            # 2. Check if a fault code has been mentioned in the chat
            with open("backend/data/fault_code_database.json", "r") as file:
                data = json.load(file)

            faultCodes = []
            # Extract the existing fault codes from the fault code database
            for faultCode in data:
                # print("fault code: ", faultCode)
                faultCodes.append(faultCode)

            # print("existing faultCodes: \n", faultCodes)

            # 2. Call the fault code model to detect a fault code in the chat
            faultCode, possible_reasons, activities = decisionTree.heardfaultCodeModel(chat, faultCodes)

            return liveSummary, infoAdded, faultCode, possible_reasons, activities


# decisionTree.decisionTree()