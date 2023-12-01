import os
import openai
from colorama import Fore
import time

########################               Functions used               ############################

def open_log():
    t = time.localtime()
    current_time = time.strftime("%H-%M-%S", t)
    log_file = 'log_from_' + current_time + '.txt'
    with open(log_file, 'w') as f:
        f.write("LOG from run starting: " + current_time)
        f.write("\n")
    f.close()
    return log_file

def log(log_file, message):
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    with open(log_file, 'a') as f:
        f.write(current_time + " [LOG: " + message + "]\n")
    f.close()

def generate_response(prompt):
        model_engine = "text-davinci-003"
        prompt = (f"{prompt}")

        completions = openai.Completion.create(
            engine=model_engine,
            prompt=prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        )

        message = completions.choices[0].text
        log("completion generated...")
        return message.strip()

def sentenceCheck(inputSentence):
    aiSentenceCheck = generate_response("Is " + inputSentence + " a sentence?")
    charList = list(aiSentenceCheck)
    if (charList[0]+charList[1]).lower() == "no":
        isSentence = False
    elif (charList[0]+charList[1]+charList[2]).lower() == "yes":
        isSentence = True
    else:
        print(Fore.RED + "LOG: [ sentenceCheckResponse=" + 
        (charList[0]+charList[1]+charList[2]).lower() +" ]" + Fore.WHITE)
        isSentence = False
    return isSentence

def process_response(response):
    splitResponse = response.split("\n", maxsplit=1)
    if len(splitResponse) == 1:
        log("responseArgs=1\n")
        # print(Fore.RED + "LOG: [ responseArgs=1 ]" + Fore.WHITE)

        print("Pepper:", response)
    elif len(splitResponse) == 2:
        log("responseArgs=2")
        # print(Fore.RED + "LOG: [ responseArgs=2 ]" + Fore.WHITE)

        isSentence = sentenceCheck(splitResponse[0])
        if isSentence == True:
            log("isSentence=True")
            # print(Fore.RED + "LOG: [ isSentence=True ]" + Fore.WHITE)

            print("Pepper:", response)
        elif isSentence == False:
            log("isSentence=False")
            # print(Fore.RED + "LOG: [ isSentence=False ]" + Fore.WHITE)

            print("Pepper: I do not understand. Can you please repeat what you said?")

#################################################################################################

api_key = os.getenv("OPENAI_API_KEY")

while True:
    speech_input = input("Speech input: ") # use naoqi to 'listen' and convert speech to text
    if speech_input.lower() == "exit": # only used for cli
        break
    log("speech_input='" + speech_input + "'")

    ###############################################################################################

    # takes the place of calling from google cloud in this script
    openai.api_key = api_key

    initial_response = generate_response(speech_input)
    process_response(initial_response)

    ###############################################################################################

    

