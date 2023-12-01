from full_query import full_query
from insert_memory import insert_memory
import openai
from dotenv import load_dotenv
import os
import yaml
from texttospeech import *
from live2D import *
import random

#TODO NEED TO IMPLEMENT MEMORY

def read_file(path_to_file):
    with open(path_to_file) as f:
        contents = ' '.join(f.readlines())
        return contents
    
def read_yaml(parameter, var):
    with open("config.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        return(cfg[parameter][var])

def key_listener(stop_event):
    while True:
        if input() == 'q':
            print("Ending Stream")
            stop_event.set()  # Signal the loop to stop
            break

def read_data():
    with open("shared.txt", "r") as file:
        data = file.read()
        return data

def get_current_action():
    with open("currentAction.txt", "r") as file:
        data = file.read()
        return data
    
    
def parseAndPlay(response):

    total = response['choices'][0]['message']['content']
    response_text = total.split("\n")[0]
    function = total.split("\n")[-1].strip().lower()
    ans = response_text
    print(ans)
    print("Sending function:" + function)
    if function.lower() in ["donothing()","confused():","hearts()", "angry()", "pentablet()", "noheadband()", "blush()", "blankeyes()", "pout()", "writetablet()", "brighteyes()"]:
        send_expression(function)
    rate = 150 + int(len(ans) * .01)
    playTTS(ans, rate)



def questionChat(questions):
    #Maybe also add types of questions like, questions about self, questions about chat, questions about streamers, questions about news
    #Need to acutally implement question saving
    keynotes = read_file("keynotes.txt")
    functions = read_file("functions.txt")
    currentAction = get_current_action()
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_API_KEY')

    prompt = "You are a vtuber with these characteristics and backstory: " + ' '.join(keynotes) + "You are currently: " + currentAction + ". You've already asked these questions: " + '?'.join(questions) + "Write an interesting question in first person you have not asked yet to your chat. No swearing or controversy. You have this set of abilities that are encoded as parameters: " + ' '.join(functions) + ". If you call a function, you will perform the action that it describes. Each function is separated from its description by a ':' and separated from other functions by a ';' After categorizing your response, simply call one function using its name and '()' and write it after a new line no punctuation."
        

    response = openai.ChatCompletion.create(
    model="gpt-4",
    messages= [{"role": "user", "content": prompt}]
    )

    parseAndPlay(response)

    


def questionFromChat(text):
    keynotes = read_file("keynotes.txt")
    functions = read_file("functions.txt")
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_API_KEY')
    currentAction = get_current_action()
    prompt = "You are a vtuber with these characteristics and backstory: " + ' '.join(keynotes)  + "You are currently: " + currentAction + ". Someone wrote this to you in chat. It may contain Twitch emotes: " + text + "If what they said is not empty or just spaces, write up a response, comment, question, or sarcastic quip about it. No emojis. ASCII characters only. No swearing or controversy. You have this set of abilities that are encoded as parameters: " + ' '.join(functions) + ". If you call a function, you will perform the action that it describes. Each function is separated from its description by a ':' and separated from other functions by a ';' After categorizing your response, simply call one function using its name and '()' and write it after a new line no punctuation."
        

    response = openai.ChatCompletion.create(
    model="gpt-4",
    messages= [{"role": "user", "content": prompt}]
    )

    parseAndPlay(response)


def generateConversation():
    keynotes = read_file("keynotes.txt")
    functions = read_file("functions.txt")
    currentAction = get_current_action()
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_API_KEY')

    prompt = "You are currently: " + currentAction + "Muse to yourself under 50 words. No swearing or controversy. It can be random. You have this set of abilities that are encoded as parameters: " + ' '.join(functions) + ". If you call a function, you will perform the action that it describes. Each function is separated from its description by a ':' and separated from other functions by a ';' After categorizing your response, simply call one function using its name and '()' and write it after a new line no punctuation."
        

    response = openai.ChatCompletion.create(
    model="gpt-4",
    messages= [{"role": "user", "content": prompt}]
    )

    parseAndPlay(response)

def generateJoke():
    keynotes = read_file("keynotes.txt")
    functions = read_file("functions.txt")

    load_dotenv()
    openai.api_key = os.getenv('OPENAI_API_KEY')

    prompt = "Give a random joke. Say it outloud in its entirety. Don't ask why don't scientists trust atoms. Make sure to add the punchline after you say the joke. No swearing or controversy. Finish the joke. You have this set of abilities that are encoded as parameters: " + ' '.join(functions) + ". If you call a function, you will perform the action that it describes. Each function is separated from its description by a ':' and separated from other functions by a ';' After categorizing your response, simply call one function using its name and '()' and write it after a new line no punctuation. "
        

    response = openai.ChatCompletion.create(
    model="gpt-4",
    messages= [{"role": "user", "content": prompt}]
    )

    parseAndPlay(response)

def generateSelfTalk():
    keynotes = read_file("keynotes.txt")
    functions = read_file("functions.txt")
    backstory = read_file("backstory.txt")
    currentAction = get_current_action()
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_API_KEY')

    prompt = "You are a vtuber with these characteristics: " + ' '.join(keynotes) + "This is your backstory: " + ' '.join(backstory)  + "You are currently: " + currentAction + ". Reminisce on a made-up story from the past under 60 words. No swearing or controversy. You have this set of abilities that are encoded as parameters: " + ' '.join(functions) + ". If you call a function, you will perform the action that it describes. Each function is separated from its description by a ':' and separated from other functions by a ';' After categorizing your response, simply call one function using its name and '()' and write it after a new line no punctuation."
        

    response = openai.ChatCompletion.create(
    model="gpt-4",
    messages= [{"role": "user", "content": prompt}]
    )

    parseAndPlay(response)

def emote():
    print("emoting")

def sayGoodbye():
    keynotes = read_file("keynotes.txt")
    functions = read_file("functions.txt")
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_API_KEY')

    prompt = "You are a vtuber with these characteristics and backstory: " + ' '.join(keynotes) + ". Tell your stream you have to go and thank them for watching the stream. No swearing or controversy. You have this set of abilities that are encoded as parameters: " + ' '.join(functions) + ". If you call a function, you will perform the action that it describes. Each function is separated from its description by a ':' and separated from other functions by a ';' After categorizing your response, simply call one function using its name and '()' and write it after a new line no punctuation."
        

    response = openai.ChatCompletion.create(
    model="gpt-4",
    messages= [{"role": "user", "content": prompt}]
    )

    parseAndPlay(response)


def sayGoodbye():
    keynotes = read_file("keynotes.txt")
    functions = read_file("functions.txt")
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_API_KEY')

    prompt = "You are a vtuber with these characteristics and backstory: " + ' '.join(keynotes) + ". Tell your stream you have to go and thank them for watching the stream. No swearing or controversy. You have this set of abilities that are encoded as parameters: " + ' '.join(functions) + ". If you call a function, you will perform the action that it describes. Each function is separated from its description by a ':' and separated from other functions by a ';' After categorizing your response, simply call one function using its name and '()' and write it after a new line no punctuation."
        

    response = openai.ChatCompletion.create(
    model="gpt-4",
    messages= [{"role": "user", "content": prompt}]
    )

    parseAndPlay(response)



def startStream(plans):
    keynotes = read_file("keynotes.txt")
    functions = read_file("functions.txt")
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_API_KEY')

    prompt = "You are a vtuber with these characteristics and backstory: " + ' '.join(keynotes) + ". You are starting your stream. Welcome chatters to your stream. Talk about your plans for the day and the future: " + plans + " No swearing or controversy. You have this set of abilities that are encoded as parameters: " + ' '.join(functions) + ". If you call a function, you will perform the action that it describes. Each function is separated from its description by a ':' and separated from other functions by a ';' After categorizing your response, simply call one function using its name and '()' and write it after a new line no punctuation."
        

    response = openai.ChatCompletion.create(
    model="gpt-4",
    messages= [{"role": "user", "content": prompt}]
    )
    
    
    parseAndPlay(response)
 