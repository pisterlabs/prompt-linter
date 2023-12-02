import openai
import configparser

config = configparser.ConfigParser()
config.read('/Users/ethanvosburg/Documents/git/ENGR-220-MatLab/FinalProject/env.ini')



def getResponse(prompt, name, probNum, assign, toggle):
    openai.api_key = config['KEYS']['OPENAI_API_KEY']
    
    if toggle == "PatGPT":
        decorationText = "Write a matlab script to solve the following problem:"
        promptIn = decorationText + prompt
    else:
        promptIn = prompt
        
    promptIn = promptIn.replace("\n", "")
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": promptIn}])
    reply = completion.choices[0].message.content

    # decorCode = open(r"DecoratedCode.txt", "w+")
    
    if toggle == "PatGPT":
        appendStr = "%% Assignment " + str(assign) + " ENGR 220\n% " + name + "\n%% Problem #" + str(probNum) + "\n% "
        genCode = open(r"GeneratedCode.m", "w+")
        genCode.write(appendStr)
        genCode = open(r"GeneratedCode.m", "a")
        genCode.write(prompt)
        genCode.write(reply)
        genCode.close()
        toMatLab = appendStr + prompt + reply
    else:
        toMatLab = reply

    return toMatLab

reply = getResponse(promptIn, nameIn, probNumIn, assignIn, toggleIn)