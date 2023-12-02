from tkinter import *
from conscious_memory import *
from talk import *
import openai
from deep_momory import *

def ai(input):
    #Check if exixsts in deep memory
    check = find(input)
    if(check == "Dismiss"):
        print("DEEP MEMORY: "+bcolors.OKGREEN+"Alright"+bcolors.ENDC)
        return "Alright"
    elif(check):
        print("DEEP MEMORY: "+bcolors.OKGREEN+check+bcolors.ENDC)
        return check
    try :
        # Call the API key under your account (in a secure way)
        openai.api_key = "ADD_YOUR_OPEN_AI_API_KEY"
        response = openai.Completion.create(
        engine="text-davinci-002",
        prompt =  input,
        temperature = 0.6,
        top_p = 1,
        max_tokens = 64,
        frequency_penalty = 0,
        presence_penalty = 0
        )
        answer = str(response.choices[0].text)
        add(input, answer)
        print("Engine: "+bcolors.OKGREEN+answer+bcolors.ENDC)
        return answer
    except:
        print("AI: "+bcolors.WARNING+"Can't get info, API call failed"+bcolors.ENDC)
        return "Can't help with that at this moment"
    
def brain(input, window):
    if(input == "hi" or input == "hello" or input == "hey" or input == "hai"):
            v = memory('greetings', 'hello')
            talk(v)
            print("CONSCIOUS MEMORY: "+bcolors.OKGREEN+v+bcolors.ENDC)
            return 'Greeting Responsed'
    elif(input == "exit please"
            or input == "shutdown please"
            or input == "go to sleep friday"
            or input == "go to sleep"):
            talk(memory('statement', 'exit'))
            window.destroy()
            print("Engine: "+bcolors.OKGREEN+"Aborted(Command)"+bcolors.ENDC)
            return 0
    elif(input == "friday chnage engine pin"
        or input == "change engine pin"):
            talk("Enter your old pin code then new pin code")
            return  'ENGINE_CHANGE_PIN'
    else :
            #Simply ask ai to do it
            talk(ai(input))
            return 'Resolve done'
