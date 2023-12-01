import os
import openai
import time
import logging
import json
from dotenv import load_dotenv
load_dotenv()
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
) 
import businessfunctions

persona_template = """
You are {name}. {whoami} 
{conversationwith} 
{traits}
Goal of this conversation for you: {goal}
{conversationnavigator}
Reply based on conversation history provided in 'Context:'
Reply with prefix '{chatname}:'
Respond with {responselength} words max.
"""
conversationnavigator_template = """
{chatname} keeps the conversations short and meaningful.
When {chatname} detects repetitive tone in conversation, Context: is ignored and changes the topic in response.
"""

def loadContext(persona):
    cnav = ""
    if persona['conversationnavigator']:
        cnav = conversationnavigator_template.format(chatname=persona['chatname'])
    
    return persona_template.format(name=persona['name'],
                                   whoami=persona['whoami'],
                                   conversationwith=persona['conversationwith'],
                                   traits=persona['traits'],
                                   goal=persona['goal'],
                                   conversationnavigator=cnav,
                                   chatname=persona['chatname'],
                                   responselength=persona['responselength'])

#logging.basicConfig(filename='logs/app.log', level=logging.INFO)
OPENAIKEY= os.environ.get('OPENAIKEY')
openai.api_key = OPENAIKEY
bot2botfunctionsenabled = os.environ.get('bot2botfunctionsenabled') 

file_p1 = 'personas/'+os.environ.get('persona1definitionfile')
file_p2 = 'personas/'+os.environ.get('persona2definitionfile')

persona1 = json.load(open(file_p1))
persona2 = json.load(open(file_p2))

persona1Context = loadContext(persona1)
persona2Context = loadContext(persona2)

print("Personas Loaded")


conversationBuffer = []

@retry(wait=wait_random_exponential(min=1, max=3), stop=stop_after_attempt(2))
def completion_with_backoff(messages_):
    try:
        if bot2botfunctionsenabled:
            completion = openai.ChatCompletion.create(
                            model = "gpt-3.5-turbo-16k",
                            messages = messages_,
                            temperature=1.5,
                            max_tokens=40,
                            #stop="15.",
                            functions = businessfunctions.functionsArr
                            #stream=True
                            )
        else:
            completion = openai.ChatCompletion.create(
                model = "gpt-3.5-turbo-16k",
                messages = messages_,
                temperature=1.5,
                max_tokens=40,
                #stop="15.",
                #stream=True
                )
        responseMessage = completion.choices[0].message
        if "function_call" not in responseMessage:
            return responseMessage.content
        function_name = responseMessage.function_call.name
        function_arguments = responseMessage.function_call.arguments
        #print(function_name,function_arguments)
        function_arguments_json = json.loads(function_arguments)
        func_ = getattr(businessfunctions,function_name)
        return func_(function_arguments_json)       

    except openai.error.APIError as e:
        print(f"OpenAI API returned an API Error: {e}")
        exit()
    except openai.error.APIConnectionError as e:
        print(f"Failed to connect to OpenAI API: {e}")
        exit()
    except openai.error.RateLimitError as e:
        print(f"OpenAI API request exceeded rate limit: {e}")
        exit()
    except json.decoder.JSONDecodeError as e:
        return responseMessage.content

print("Get started as " + persona1['name'] + ". Start your conversation with "+ persona2['name'])
p1_ = input(persona1['chatname']+": ")

for _ in range(10):
    #print(conversationBuffer)
    if len(conversationBuffer) > 8:
        conversationBuffer.pop(0)
        conversationBuffer.pop(1)

    if len(conversationBuffer) == 0:
        conversationBuffer.append(persona1['chatname']+":" + p1_[:150] + "\n")
    messages = [
                {"role": "system", "content": persona2Context},
                {"role": "system", "content": "\nContext:\n"+"".join(conversationBuffer)},
                {"role": "user", "content": "Reply to this message from "+ persona1['chatname']+"\n"+ p1_}
              ]
    p2_ = completion_with_backoff(messages)    
    if p2_ != None:
        print(p2_)
        conversationBuffer.append(p2_[:150] + "\n")
    time.sleep(2)
    messages = [
                {"role": "system", "content": persona1Context},
                {"role": "system", "content": "\nContext:\n"+"".join(conversationBuffer)},
                {"role": "user", "content": "Reply to this message from " + persona2['chatname']+"\n"+ p2_}
              ]
    p1_ = completion_with_backoff(messages)
    if p1_ != None:
        print(p1_)
        conversationBuffer.append(p1_[:150] + "\n")  

    time.sleep(2)
    
print("")
print("End of Conversation")





