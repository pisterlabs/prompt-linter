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
Reply based on conversation history provided in 'Context:'
Reply with prefix '{chatname}:'
Respond with {responselength} words max.
"""

def loadContext(persona):
    return persona_template.format(name = persona['bot']['name'],
                                   whoami = persona['bot']['whoami'],
                                   conversationwith = persona['bot']['conversationwith'],
                                   traits = persona['bot']['traits'],
                                   goal=persona['bot']['goal'],
                                   chatname = persona['bot']['chatname'],
                                   responselength = persona['bot']['responselength'])

#logging.basicConfig(filename='logs/app.log', level=logging.INFO)
OPENAIKEY= os.environ.get('OPENAIKEY')
openai.api_key = OPENAIKEY

file_b = 'personas/'+os.environ.get('botdefinitionfile')
thebot = json.load(open(file_b))
botcontext = loadContext(thebot)
conversationBuffer = []

@retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(2))
def completion_with_backoff(messages_):
    try:
        completion = openai.ChatCompletion.create(
                           model = "gpt-3.5-turbo-16k",
                           messages = messages_,
                           temperature=1.5,
                           max_tokens=50,
                           stop="15.",
                           functions = businessfunctions.functionsArr
                           #stream=True
                           )
        responseMessage = completion.choices[0].message
        if "function_call" not in responseMessage:
            return responseMessage.content

        function_name = responseMessage.function_call.name
        function_arguments = responseMessage.function_call.arguments
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


print("Conversing as " + thebot['human']['name'] + ". Start your conversation with bot "+ thebot['bot']['name'] )

for _ in range(8):
    print("_")
    if len(conversationBuffer) > 10:
        conversationBuffer.pop(0)
        conversationBuffer.pop(1)
    p1_ = input(thebot['human']['chatname']+": ")
    messages = [
                {"role": "system", "content": botcontext},
                {"role": "system", "content": "Context:\n"+"".join(conversationBuffer)},
                {"role": "user", "content": thebot['human']['chatname']+":"+ p1_}
              ]
    p2_ = completion_with_backoff(messages)
    print(p2_)
    conversationBuffer.append(thebot['bot']['chatname']+":" + p1_[:150] + "\n")
    conversationBuffer.append(p2_[:150] + "\n")
    time.sleep(3)
print("End of Conversation")





