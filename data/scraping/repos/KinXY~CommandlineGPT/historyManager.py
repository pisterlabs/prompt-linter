import openai
import json
import os

import prompts
import chat
import openai_parameters

openai.api_key = json.loads(open("./config.json", "r").read())["api_key"]

def chooseHistory(myChat):
    # open the json file of all history
    with open("./history.json", "r") as f:
        historyList = json.loads(f.read())
    # print all the history with summary, with index
    historyIndex = 1
    for i in historyList:
        print(str(historyIndex) + i["file"] + " : " + i["summary"])
        historyIndex += 1
    print()
    # ask the user to choose a history
    while True:
        print("Please enter the index of the history you want to load: ")
        historyIndex = int(input("Input 0 if you want to start a new chat:"))
        if historyIndex > len(historyList) or historyIndex < 0:
            print("\033[91m" + "Invalid index!" + "\033[0m")
        else:
            break
    # if the user want to start a new chat
    if historyIndex == 0:
        return
    else:
        loadDialogue(myChat, historyList[historyIndex - 1]["file"])

def loadDialogue(myChat, title):
    myChat.title = title
    with open(title, "r") as f:
        listDialogue = json.loads(f.read())
    myChat.chatHead = chat.dictToTree(listDialogue)
    myChat.refreshEnd()
    myChat.refreshHistory()

def summarizeHistory(myChat):
    myHistory = myChat.history.copy()
    myHistory.append({"role": "user", "content": prompts.summary_prompt})
    try:
        summary = openai.ChatCompletion.create(
                model = "gpt-3.5-turbo",
                messages = myHistory,
                max_tokens = openai_parameters.max_tokens,
                temperature = openai_parameters.temperature,
            )
    except:
        print("\033[91m" + "OpenAI API Error!\nUse previous summary." + "\033[0m")
        return ""
    return summary.choices[0].message.content

def dumpHistory(myChat):
    if os.path.getsize("./history.json") == 0:
        nowHistory = []
    else:
        with open("./history.json", "r") as f:
            nowHistory = json.loads(f.read())
    # delete the original json file
    summaryBackup = ""
    if myChat.title != "":
        os.system("rm " + myChat.title)
        for i in range(len(nowHistory)):
            if nowHistory[i]["file"] == myChat.title:
                summaryBackup = nowHistory[i]["summary"]
                nowHistory.pop(i)
                break
    # dumps the history to a json file
    myChat.dumpHistory()
    # summarize the history
    summary = summarizeHistory(myChat)
    if summary == "":
        if summaryBackup != "":
            summary = summaryBackup
        else:
            summary = "No summary"
    # update the history.json
    nowHistory.append({"file" : myChat.title, "summary" : summary})
    with open("./history.json", "w") as f:
        f.write(json.dumps(nowHistory))
