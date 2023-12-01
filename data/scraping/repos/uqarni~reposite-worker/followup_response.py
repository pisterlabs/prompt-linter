import json
import os
import time

import openai
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("API_KEY")
openai.api_key = api_key

def followup(yesno):
    if yesno == 'yes':
        return 'y'
    if yesno == 'no':
        return 'n'
    else:
        return 'n'



########
functions=[
            {
                "name": "followup",
                "description": "Analyze the conversation and determine a follow-up message should be sent based on Taylor's last message",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "yesno": {
                            "type": "string",
                            "description": "yes if a response is warranted, and no if it is not",
                            "enum": ['yes', 'no']
                        }
                    },
                    "required": ["yesno"],
                },
            }
        ]



# Step 1, send model the user query and what functions it has access to
def run_conversation(bot_messages: list):

    prompt = '''
    Hello Boss Can you help our assistant Taylor who handles lead responses for Reposite?
    Taylor typically responds to leads and avoids answering to ones who show disinterest or ask not to contact them. Your task will be to analyze the whole conversation and determine whether a follow-up message should be sent or not based on Taylor's last message and the whole conversation.
    If you determine a follow-up message should be sent based on Taylor's last message in the conversation, respond with "yes." If you determine a follow-up message is not necessary, respond with "no.". Here are some examples with output.
    Example 1:
        Assistant: I completely understand, uzair. I'll make sure you don't receive any more messages
        about this.
        Should you change your mind and decide to explore building a more consistent pipeline of
        group business, feel free to reach out or check out the membership benefits
        here: https://www.reposite.io/membership-overview-1. Have a great day!
        Lead: I said to stop messaging me
        Assistant: I apologize if my messages have been intrusive, uzair. I respect your request and
        will refrain from sending further communications. Should you have any change of mind about
        our offerings at Reposite, you can always check out the membership details
        here: https://www.reposite.io/membership-overview-1. Wishing you the best.
        output = no
    Example 2:
        Lead: nevermind I hate you stop emailing me
        Assistant: I'm sorry to hear that, Uzair. I will ensure you don't receive further communications
        from me.
        Should you change your heart, our membership information is always accessible on our website.
        My sincerest apologies for any inconvenience caused.
        Lead: I said stop
        Lead: Actually, I would like to know more. Can I stay on the free tier or do I have to pay the
        membership?
        Assistant: Absolutely, Uzair! You can certainly stay on the Reposite Basic plan. However, I would
        like to draw attention to the fact that with this plan, you will only be able to receive ad hoc
        direct inbound requests and won't have a convenient level of transparency or abilities to
        promote
        your services.
        output = yes

    '''

    
    all_messages = [{'role': 'system', 'content': prompt}]
    
    #for iterate through redis database of all the conversations thus far:
    all_messages.append({"role": "user", "content": f'''Hey Boss - here's my message history so far. I am the assistant. 
    Please review the last message of the Taylor in the conversation and advise me whether I should send followup? Here's my 
    conversation.\n\n{bot_messages}'''})

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            temperature = 0,
            messages=all_messages,
            functions=functions,
            function_call= {"name": "followup"},
            user='reposite'
        )
        if response:
            message = response["choices"][0]["message"]

            # Step 2, check if the model wants to call a function
            if message.get("function_call"):
                followup_check = json.loads(message["function_call"]["arguments"])["yesno"]
                return followup_check
            else:
                return 'error'
    except Exception as e:
        print(f"An error occurred: {e}")

