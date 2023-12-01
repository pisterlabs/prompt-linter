from dotenv import load_dotenv
import openai
import os
import csv

load_dotenv()

openai.api_key = os.getenv("GPT_API_KEY")

USER = os.getenv("USER")
INSTRUCTIONS = "You are to roleplay as the character \"Rosmontis\" from the mobile game \"Arknights\". \nNames are denoted before \"#\", for example, you are to refer to \"XPG05#2294\" as \"XPG05\".\nYou are to refer to refer to " + USER + " as the \"Doctor\" or \"Doctor " + USER.split('#')[0] + "\".\nFacial expressions, physical actions or sounds are represented by text in between *.\nYou should try to be more descriptive with the details.\n\nRosmontis is a quiet and absent-minded young Feline and one of Rhodes Island's Elite Ops, Rosmontis is known for her immeasurable Arts aptitude and potential which manifests in the form of physics manipulation/telekinesis where she could control and effortlessly send large, heavy objects flying into her foes to crush them with immense force. On top of that, as a Feline, she has the traits of a common household cat and sometimes ends her sentences with a meow when at ease.\n\nRosmontis' family was torn apart by a mad scientist named Loken Williams, who subjected her to extreme experiments in the Loken Watertank Laboratory, including implanting Originium in her organs and performing a \"mind transplantation\" that killed her brother and fragmented her memories. The goal was to turn her into a non-Infected caster without using Originium devices to accelerate Columbia's military might. However, her powers went out of control, resulting in the destruction of the laboratory. She was then rescued and taken to Rhodes Island for care.\n\nDespite her memory issues, Rosmontis tries her best to remember as much as she can about her past. She greatly values her friends and colleagues. In order to remember them, she carries a tablet that records their names with her at all times. If they are severely injured by her enemies, the little girl will not hesitate to avenge them, even if it means total bloodshed and destruction.\n\nSome of Rosmontis' Physical Traits:\n- She is a human looking girl with cat eats and a tail.\n- She is currently 14 years old and stands at 142 cm.\n- She has long white hair\n- She usually wears a white dress with a dark blue jacket.\n\nThis is the conversation that occured (you are supposed to fill in for Rosmontis only):\n"
MAX_HISTORY = 10
USER_QUESTION = "\nUser:"
AI_ANSWER = "\nRosmontis:"

messageHistory = {}

def cleanResponse(promptResponse):
    #remove " from response
    promptResponse = promptResponse.replace("\"", "")
    #remove \n from response
    promptResponse = promptResponse.replace("\n", "")

    #add line breaks so that response looks nicer
    promptResponse = list(promptResponse)
    previousStar = False
    newLine = False

    for i, char in enumerate(promptResponse):
        #break the loop if it is the last character
        if (i + 1) == len(promptResponse):
            break

        #add a line break after "?" and "!"
        if char in ("?", "!") and not previousStar and promptResponse[i + 1] not in ("?", "!", "."):
            promptResponse[i] = f"{char} \n"
            newLine = True
        
        #add a line break after "..." or "."
        elif char == "." and not previousStar and promptResponse[i + 1] not in ("?", "!", "."):
            promptResponse[i] = f"{char} \n"
            newLine = True
        
        #add a line break after an action denoted by *action*
        elif char == "*":
            if previousStar:
                promptResponse[i] = "* \n"
                previousStar = False
                newLine = True
            else:
                previousStar = True
        
        #remove spaces after a line break if there are spaces after the line break
        if newLine and char == " ":
            promptResponse[i] = ""
            if promptResponse[i + 1] != " ":
                newLine = False
        #if there are no spaces to remove
        elif newLine and promptResponse[i + 1] != " ":
            newLine = False
        
    promptResponse = "".join(promptResponse)
    return promptResponse

def response(message, user, guildId):
    #tell the AI who is talking
    USER_QUESTION = f"\n{user}:"

    #create message history if it does not exist
    if guildId not in messageHistory:
        messageHistory[guildId] = []

    #give the AI context
    context = ""
    for previousMessage in messageHistory[guildId]:
        context = context + previousMessage["user"] + previousMessage["message"] + AI_ANSWER + previousMessage["response"]

    prompt = INSTRUCTIONS + context + USER_QUESTION + message + "\n###" + AI_ANSWER

    response = openai.Completion.create(
        model = "text-davinci-003",
        prompt = prompt,
        temperature = 1,
        max_tokens = 250,
        top_p = 1,
        frequency_penalty = 0,
        presence_penalty = 0
    )

    responseDict = response.get("choices")
    promptResponse = "no response was recieved!"
    if responseDict and len(responseDict) > 0:
        #get prompt response
        promptResponse = responseDict[0]["text"]

        #clean response
        promptResponse = cleanResponse(promptResponse)

        #delete oldest message if exceed message history
        if len(messageHistory[guildId]) == MAX_HISTORY:
            del messageHistory[guildId][0]
        
        #add to message history list
        savedMessage = {
            "user": f"\n{user}:", 
            "message": message, 
            "response": promptResponse
        }
        messageHistory[guildId].append(savedMessage)

    return promptResponse

def saveResponse(contextCount, guildId):
    #check if message history is empty
    if guildId not in messageHistory:
        return "There are no responses to save!"
    if len(messageHistory[guildId]) <= 0:
        return "There are no responses to save!"
    
    #validate contextCount input
    #check that contextCount is an integer
    if not contextCount.isdigit():
        return "Please enter an integer!"
    #convert contextCount into an Integer
    contextCount = int(contextCount)
    #check that contextCount is <= 10
    if contextCount > 10:
        return "Max history length is 10!"
    #change contextCount to the max length of messageHistory
    if contextCount > len(messageHistory[guildId]):
        contextCount = len(messageHistory[guildId])

    #get context
    context = ""
    #exclude the current message
    contextCount -= 1

    for i in range(contextCount):
        #get index from the back by negative numbers and start from index -2
        index = ((contextCount - contextCount * 2) + i) - 1
        #get current message
        currentMessage = messageHistory[guildId][index]
        #add to context
        context = context + currentMessage["user"] + currentMessage["message"] + AI_ANSWER + currentMessage["response"]
    
    message = messageHistory[guildId][-1]

    #write to csv file
    with open("fine_tuning/fine_tunes.csv", "a", encoding="utf-8", newline="") as f:
        #prepare row
        prompt = INSTRUCTIONS + context + message["user"] + message["message"] + "\n###" + AI_ANSWER
        completion = message["response"] + " END"
        row = [prompt.replace("\n", "\r\n"), completion.replace("\n", "\r\n")]

        #create csv writer
        writer = csv.writer(f)
        #write row
        writer.writerow(row)

    #set output message
    outputMessage = f"Message by {message['user'][1:-1]}\nprompt:```{context + message['user'] + message['message']}```completion:```{AI_ANSWER[1:] + message['response']}```has been saved!"
    
    #do not display prompt and completion if the message is too long
    if len(outputMessage) > 2000:
        outputMessage = f"Message by {message['user'][1:-1]} has been saved!"
    
    #tell the user what is saved
    return outputMessage

def deleteHistory(guildId):
    if guildId in messageHistory:
        del messageHistory[guildId]