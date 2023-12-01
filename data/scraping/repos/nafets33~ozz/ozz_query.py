import json
import os
import openai
from dotenv import load_dotenv
# from master_ozz.utils import ozz_master_root
# Loading environment variables
load_dotenv('.env')

# Loading the json common phrases file and setting up the json file
json_file = open('fastapi/greetings.json','r')
common_phrases = json.load(json_file)

# Setting up the llm for conversation with conversation history
def llm_assistant_response(message,conversation_history):
        conversation_history.append({"role": "user", "content": message})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation_history,
            api_key=os.getenv('ozz_api_key')
        )
        assistant_reply = response.choices[0].message["content"]
        return assistant_reply


# call_llm=True # goal is to set it to False and figure action/response using local phrases as required
# Now we are only using llm when we don't have response to the query in greetings.json
def Scenarios(current_query : str , conversation_history : list , first_ask=True, conv_history=False):
    if first_ask:
        ''' Appending the prompt for system when user asks for first time (is this first ask?) 
        also with json coz if user again tries to ask something and doesn't found in json then it will go to llm
        so llm needs to be already have the json conversation to understand the next query asked by user '''

        conversation_history.append({"role": "system", "content": "You are a cute and smart assistant for kids."})

        # For first we will always check if anything user asked is like common phrases and present in our local json file then give response to that particular query
        for query, response in common_phrases.items():
            if query in current_query.lower():
                # Appending the user question from json file
                conversation_history.clear() if not conv_history else conversation_history.append({"role": "user", "content": current_query})
                # Appending the response from json file
                conversation_history.clear() if not conv_history else conversation_history.append({"role": "assistant", "content": response})
                return response 
        
        else:
            ############## This code needs to run when the response is not present in the predefined json data ################
            # Appending the user question
            # conversation_history.clear() if not conv_history else conversation_history.append({"role": "user", "content": current_query})
            # Calling the llm
            assistant_response = llm_assistant_response(current_query,conversation_history)
            # assistant_response = 'thanks from llm'
            # Appending the response by llm
            conversation_history.clear() if not conv_history else conversation_history.append({"role": "assistant", "content": assistant_response})
            return assistant_response 

    # This is the case when first_ask is already done or user already have asked a query with llm 
    else:
        # For first we will always check if anything user asked is like common phrases and present in our local json file then give response to that particular query
        for query, response in common_phrases.items():
            if query in current_query.lower():
                # Appending the user question from json file
                conversation_history.clear() if not conv_history else conversation_history.append({"role": "user", "content": current_query})
                # Appending the response from json file
                conversation_history.clear() if not conv_history else conversation_history.append({"role": "assistant", "content": response})
                return response 
        
        else:
            ############## This code needs to run when the response is not present in the predefined json data ################
            # Appending the user question
            # conversation_history.clear() if not conv_history else conversation_history.append({"role": "user", "content": current_query})
            # Calling the llm
            assistant_response = llm_assistant_response(current_query,conversation_history)
            # assistant_response = 'thanks from llm'
            # Appending the response by llm
            conversation_history.clear() if not conv_history else conversation_history.append({"role": "assistant", "content": assistant_response})
            return assistant_response 

# Testing the functions    
# conversation_history = []
# print(Scenarios('hello buddy',conversation_history))
# print(conversation_history)