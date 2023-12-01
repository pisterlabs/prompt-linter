import json
import os
import requests
import io
import base64
from PIL import Image, PngImagePlugin
from random import choice
from datetime import datetime
import openai
from typing import Final
from telegram import Update
import toolbox
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

import os

def loadSystemParameters():
    params = ["TELEGRAM_API_KEY", "OPENAI_API_KEY"]
    args = {}
    
    # Find global environment variables first. If not found, try the .env file
    for param in params:
        args[param] = os.environ.get(param)

    # Check if there are missing values from the global env. If so, load the local .env file
    if None in args.values() or '' in args.values():
        loadEnv()

    # Validates and informs user what is missing
    flag = False
    for param in params:
        if os.environ.get(param) == '':
            print("Please enter in a value for " + param)
            flag = True
    
    if flag: exit(1)

def loadEnv(dotenv_path=".env"):
    """Load key-value pairs from a .env file into the environment."""
    if not os.path.exists(dotenv_path):
        raise ValueError(f"Please rename example.env -> .env: {dotenv_path}")

    with open(dotenv_path, "r") as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith("#"):  # Ignore comments and empty lines
                key, value = line.split("=", 1)
                value = value.replace("\"", "").strip()
                if value == "":
                    continue # Skip empty values or existing environment variables
                os.environ[key] = value

# To test the function
# loadSystemParameters()

loadSystemParameters()
TOKEN = os.environ.get("TELEGRAM_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ADMIN_ID = 6146500807
USER_ID = ADMIN_ID
BOT_USERNAME = "@.bot"
OPENAI_MODEL = "gpt-3.5-turbo"
defaultRuntimeVars = {
    "id": None,
    "userName": "Guest",
    "character": None,
    "nsfw": "false",
    "sd": "false",
    "lastMessage": None,
}
defaultUserVars = {"tasks": [], "values": [], "interests": []}
characterTraitsWeight = 1
characterStartMessageFrequency = 3
maxTokenSize = 2000

def main():
    initComm()


def buildSDPayload(userID, parameters, type="default"):
    """
    Builds the positive and negative prompts for stable diffusion.

    Gets the user's selected character's sd prompt then will replace words
    if nsfw is enabled. Then adds additional paramaters like the setting, and atmosphere
    as well as it's weight.
    """
    config = getRuntimeVars(userID)
    character = config.get("character")
    nsfwAllowed = config.get("nsfw") == "true"

    SDPrompts = getSDDefault(userID, character)
    SDNormalPrompt = SDPrompts.get("positive")
    SDNegativePrompt = SDPrompts.get("negative")
    SDExplicitPayload = SDPrompts.get("payload")

    SDPayload = None
    if SDExplicitPayload != None:
        SDPayload = getSDPayload(SDExplicitPayload)
    else:
        SDPayload = getSDPayload(type)

    SDPositivePrompt = insertSDParams(parameters, SDNormalPrompt, characterTraitsWeight)
    SDPayload["prompt"] = SDPositivePrompt
    SDPayload["negative_prompt"] = SDNegativePrompt

    return SDPayload


def updateConversation(userID, character, message):
    """
    Updates the conversation history for a given user and character.

    Parameters:
    - userID (str): The ID of the user.
    - character (str): The name of the character.
    - message (str): The message to be appended to the conversation history.
    """

    # Define the file path
    file_path = f"Data/{userID}/Characters/{character}/History.json"

    # Read the old conversation data or initialize it as an empty list if the file is empty
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            old = json.load(f)
    except json.decoder.JSONDecodeError:
        old = []

    # Append the new message to the old conversation data
    old.append(message)

    # Write the updated conversation data back to the file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(old, f, indent=4)


def reduceMemory(userID, character):
    # Selects all lines up til the fourth history line
    with open(
        f"Data/{userID}/Characters/{character}/CHPrompt.txt", "r", encoding="utf-8"
    ) as f:
        lines = f.readlines()
        keepLines = []
        for i in range(0, len(lines)):
            if "--- history ---" in lines[i].lower():
                keepLines.insert(i, lines[i])
                if len(lines) > i + 3:
                    keepLines.insert(i + 1, lines[i + 1])
                    keepLines.insert(i + 2, lines[i + 2])
                    keepLines.insert(i + 3, lines[i + 3])
                break
            keepLines.insert(i, lines[i])
    # Rewrites files with selected lines
    with open(f"{character}Prompt.txt", "w", encoding="utf-8") as f:
        f.writelines(keepLines)


def getAvailableCharacters():
    """Returns list of character names in global"""
    folderToSearch = r"Characters"
    if not os.path.exists(folderToSearch):
        return []
    files = os.listdir(folderToSearch)
    characters = [
        os.path.splitext(file)[0]
        for file in files
        if os.path.isfile(os.path.join(folderToSearch, file))
    ]
    return characters


def getFileWordCount(character):
    with open(f"{character}Prompt.txt", "r", encoding="utf-8") as infile:
        words = 0
        characters = 0
        for lineno, line in enumerate(infile, 1):
            wordslist = line.split()
            words += len(wordslist)
            # characters += sum(len(word) for word in wordslist)
        return words


def getSDDefault(id, character):
    with open(
        f"Data/{id}/Characters/{character}/Diffusion.json", "r+", encoding="utf-8"
    ) as f:
        return json.load(f)


def insertSDParams(parameters, default, weight):
    """Given character's generated parameters, default prompt string,
    and character trait weight returns string combining them all"""

    pmStr = ", ".join(parameters)
    default += f", ({pmStr}:{str(weight)})"
    return default


def getCharacterPrompt(userID, character):
    """Returns a json representation of a character"""
    if userID == 0:
        with open(f"Characters/{character}.json", "r+", encoding="utf-8") as f:
            return json.load(f)
    else:
        with open(
            f"Data/{userID}/Characters/{character}/CharacterCard.json",
            "r+",
            encoding="utf-8",
        ) as f:
            return json.load(f)

# Learned something, only use one read() function in a with open f stream
def getConversation(userID, character, chatID=None):
    """Returns a json conversation given userID, character name, and maybe, a chatID (for multiple chats)"""
    with open(
        f"Data/{userID}/Characters/{character}/History.json", "r", encoding="utf-8"
    ) as f:
        content = f.read()
        if content == "":
            return None
        else:
            return json.loads(content)


def getSystemPrompt():
    with open(f"Prompt/system.txt", "r+", encoding="utf-8") as f:
        return f.read()


def varInsert(prompt: str, replacements: dict):
    """Takes in a string containg vars like for ex. {{user}} or {{char}}
    and returns the prompt with the correlating values of parameter dict inserted
    to each occurence

    Ex. prompt="Hi, my name is {{char}}. Hey {{char}}, my name is {{user}}
        dict={"char":"joe","user":"clay"}
    returns "Hi, my name is joe. Hey joe, my name is clay

    """
    for key, value in replacements.items():
        placeholder = "{{" + key + "}}"
        prompt = prompt.replace(placeholder, value)

    return prompt


def processJsonPrompt(obj, get=""):
    """Appends specific set of vals of a dict to a str and returns new str"""

    new = ""
    sepL = "\n---"
    sepR = "---\n"
    include = {
        "Name": ["name", "char_name"],
        "Description": ["description"],
        "Scenario": ["scenario", "world_scenario"],
        "Example Chat": ["sampleChat", "example_dialogue", "mes_example"],
        "Persona": ["char_persona", "persona"],
        "Personality": ["personality"],
    }

    if get != "":
        sepL = ""
        sepR = ""
        include = {"Greeting": ["char_greeting", "greeting", "first_mes"]}

    if obj.get("spec") == "chara_card_v2":
        obj = obj["data"]

    added_titles = set()  # To keep track of titles that have been added

    for title in include:
        if title in added_titles:  # Skip if title is already added
            continue
        for prop in include[title]:
            if obj.get(prop):
                new += f"{sepL}{title}{sepR}{obj[prop]}"
                added_titles.add(title)  # Mark title as added
                break
    print(new)
    return new.strip()


def processMessageSchema(obj):
    """
    Process the given message schema to produce a list of messages.

    Args:
    - obj (dict): Dictionary containing message schema.

    Returns:
    - list: List of message dictionaries.
    """

    system_content = (
        obj["system"]["rules"]
        + "\n"
        + obj["system"]["characterDetails"]
        + "\n"
        + obj["system"]["userDetails"]
    )
    system_message = {"role": "system", "content": system_content}

    assistant_message = {
        "role": "assistant",
        "content": obj["assistant"]["firstMessage"],
    }

    user_message = {"role": "user", "content": obj["user"]}

    messages = [system_message, assistant_message]

    if obj["history"] == None:
        pass
    else:
        for entry in obj["history"]:
            message = {"role": entry["role"], "content": entry["msg"]}
        messages.append(message)

    messages.append(user_message)

    return messages  # May return messages with or without history


def sendMessage(userID, character, userMessage):
    """Takes a character name string and message as parameters. Inits the ai model. Decodes response. Returns dict if successfull"""

    config = getRuntimeVars(userID)
    user = config["userName"]

    # May want to be changed later for more complexity
    vars = {"user": user, "char": character}

    systemPrompt = getSystemPrompt()
    characterJson = getCharacterPrompt(userID, character)
    # Def needs to be changed
    userPrompt = ""

    historyJson = getConversation(userID, character)  # May be none

    messageSchema = {
        "system": {
            "rules": varInsert(systemPrompt, vars),
            "characterDetails": varInsert(processJsonPrompt(characterJson), vars),
            "userDetails": varInsert(userPrompt, vars),
        },
        "assistant": {
            "firstMessage": varInsert(
                processJsonPrompt(characterJson, get="Greeting"), vars
            )
        },
        "history": historyJson,
        "user": userMessage,
    }

    messages = processMessageSchema(messageSchema)

    availableTools = toolbox.getAvailableTools()

    toolInstances = {tool.__name__: tool() for tool in availableTools}
    toolSchemas = [instance.schema for instance in toolInstances.values()]
    functions = toolSchemas

    openai.api_key = OPENAI_API_KEY
    response = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=1,
        frequency_penalty=0.7,
        presence_penalty=0,
        top_p=1,
        functions=functions,
        function_call="auto",
    )

    replyText = ""
    secondResponse = None
    secondResponseMsg = None
    responseMsg = response["choices"][0]["message"]
    if responseMsg.get("content"):
        replyText = characterMessageClean(responseMsg.get("content"), character)

    if responseMsg.get("function_call"):
        chosenTool = toolInstances.get(responseMsg["function_call"]["name"])
        functionToCall = chosenTool.func

        functionJsonArgs = json.loads(responseMsg["function_call"]["arguments"])

        functionResponse = None
        if chosenTool.needID:
            functionResponse = functionToCall(userID, functionJsonArgs)
        else:
            functionResponse = functionToCall(functionJsonArgs)

        messages.append(responseMsg)
        messages.append(
            {
                "role": "function",
                "name": responseMsg["function_call"]["name"],
                "content": str(functionResponse),
            }
        )
        secondResponse = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=1,
            frequency_penalty=0.7,
            presence_penalty=0,
            top_p=1,
        )

    if secondResponse:
        secondResponseMsg = secondResponse["choices"][0]["message"]
        replyText = characterMessageClean(secondResponseMsg["content"], character)

    print(response)
    return replyText


def characterMessageClean(text, character):
    useTelegram = False
    if useTelegram:
        text = text.replace("*", "**")
    if text.split(character + ":")[0] == "":
        return text.split(character + ":")[1]
    else:
        return text


def getImage(payload):
    config = getRuntimeVars(USER_ID)
    url = "http://127.0.0.1:7860"
    response = requests.post(url=f"{url}/sdapi/v1/txt2img", json=payload)
    r = response.json()

    for i in r["images"]:
        image = Image.open(io.BytesIO(base64.b64decode(i.split(",", 1)[0])))
        png_payload = {"image": "data:image/png;base64," + i}
        response2 = requests.post(url=f"{url}/sdapi/v1/png-info", json=png_payload)
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("parameters", response2.json().get("info"))
        now = datetime.now()
        date = now.strftime("%m-%d-%H-%M")
        path = f"C:/Users/cw1a/AI/Fractal/Data/{USER_ID}/Characters/{config.get('character')}/output/output{date}.png"
        image.save(path, pnginfo=pnginfo)
        return path


def getTime():
    now = datetime.now()
    time_string = now.strftime("%H:%M")
    return time_string


# [1] Entry
def initComm():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("mode", mode_command))
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(MessageHandler(filters.TEXT, handleMessage))
    app.add_error_handler(error)

    print("Polling...")
    app.run_polling(poll_interval=5)


def checkUserExists(userID):
    return os.path.exists(f"Data/{userID}")


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    config = getRuntimeVars(update.message.chat.id)
    if checkUserExists(update.message.chat.id) and config.get("userName") != "Guest":
        await update.message.reply_text(
            f"Hey {config.get('userName')}, who would you like to speak to?"
        )
        setRuntimeVars(update.message.chat.id, {"character": None})
    else:
        clearRuntimeVars(update.message.chat.id)
        await update.message.reply_text("Greetings!\nWho am I speaking to?")


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    character = getRuntimeVars(update.message.chat.id).get("character")
    clearConversation(update.message.chat.id, character)
    await update.message.reply_text(f"Erased {character}'s memory :(")


async def mode_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    userID = update.message.chat.id
    if "nsfw" in text:
        setRuntimeVars(userID, {"nsfw": "true"})
        await update.message.reply_text("Ok, wow, noted.")
    elif "sfw" in text:
        setRuntimeVars(userID, {"nsfw": "false"})
        await update.message.reply_text("Good boy! :)")
    elif "sd" in text:
        setRuntimeVars(userID, {"sd": "true"})
        await update.message.reply_text(
            "Stable Diffusion mode enabled. Seperate words with a comma"
        )
    elif "chat" in text:
        setRuntimeVars(userID, {"sd": "false"})
        await update.message.reply_text("Chat mode enabled.")


def clearConversation(userID, character):
    with open(f"Data/{userID}/Characters/{character}/History.json", "w") as f:
        f.write("")


def characterSelect(text, userID):
    characters = getAvailableCharacters()
    if len(characters) == 0:
        return "No characters found"
    # Adds a character to the user's config
    if text in str(characters).lower():
        setRuntimeVars(userID, {"character": text.capitalize()})
        genCharacterVars(userID, text.capitalize(), True)
        return f"{text.capitalize()} joined..."


def processUserInit(text: str, userID):
    incoming: str = text.lower()
    config = getRuntimeVars(userID)

    # Detects users and stores their config
    if config.get("userName") == "Guest" and config.get("userName") != None:
        if len(text.split(" ")) < 2:
            setRuntimeVars(userID, {"userName": text})
            return f"Hey {text}, who would you like to speak to?"
        else:
            return "Try writing your first name, or shorter length"

    elif not config.get("userName") == "Guest":
        return characterSelect(incoming, userID)

    return "You silly, enter your name!"


# [2] Incoming messages come here
async def handleMessage(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    userID = update.message.chat.id
    USER_ID = userID
    config = getRuntimeVars(userID)
    response = ""

    canChat = config.get("userName") != "Guest" and config.get("character") != None
    print(f'User ({userID}): "{text}"')

    # User init. Handles their preferences
    if not canChat:
        response = processUserInit(text, userID)

    if canChat:
        doStableDiffusion = config.get("sd") == "true"
        setRuntimeVars(userID, {"lastMessageTime": getTime()})

        if doStableDiffusion:
            imagePath = getImage(buildSDPayload(userID, "(" + text + ")"))
            sendPhoto(imagePath, userID)

        else:
            feedback = ""
            parameters = ""

            characterMessage = sendMessage(userID, config["character"], text)
            print(f"{config['character']}: {characterMessage}")
            updateConversation(
                userID,
                config["character"],
                {"name": config["userName"], "role": "user", "msg": text},
            )
            updateConversation(
                userID,
                config["character"],
                {
                    "name": config["character"],
                    "role": "assistant",
                    "msg": characterMessage,
                },
            )

            # sendPhoto(getImage(buildSDPayload(userID, parameters)), userID)

            await update.message.reply_text(characterMessage)

    # Still choosing their config
    else:
        await update.message.reply_text(response)


# Log errors


async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"Update {update} caused error {context.error}")


def sendPhoto(file, chatID):
    url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto?chat_id={chatID}"
    img = open(file, "rb")
    requests.post(url, files={"photo": img})


def setRuntimeVars(id: int, data: dict):
    """Appends new dictionary items to existing or non-existing config"""
    old = getRuntimeVars(id)
    old.update(data)
    with open(f"Data/{id}/Runtime.json", "w") as f:
        json.dump(old, f)

    """Creates a new folder tree and associated files for a user | Overwrites runtime if called"""


def genRuntimeVars(id):
    if id == 0:
        id = 6146500807
    os.makedirs(f"Data/{id}/Characters", exist_ok=True)
    with open(f"Data/{id}/Runtime.json", "w") as f:
        data = defaultRuntimeVars
        data["id"] = id
        json.dump(data, f)
    # Will not ever overwrite User storage
    if not os.path.exists(f"Data/{id}/User.json"):
        with open(f"Data/{id}/User.json", "w") as f:
            data = defaultUserVars
            json.dump(data, f)


def genCharacterVars(
    id,
    character,
    useGlobal=False,
    overWrite=False,
    targets=["History.json", "CharacterCard.json"],
):
    newDir = f"Data/{id}/Characters/{character}"
    print(newDir)
    os.makedirs(newDir + "/Output", exist_ok=True)
    for target in targets:
        if not os.path.exists(newDir + "/" + target):
            with open(f"{newDir}/{target}", "w") as f:
                if useGlobal and target == "CharacterCard.json":
                    json.dump(getCharacterPrompt(0, character), f)
                else:
                    f.write("")
        if overWrite:
            with open(f"{newDir}/{target}", "w") as f:
                if useGlobal:
                    json.dump(getCharacterPrompt(0, character), f)
                else:
                    f.write("")


def getRuntimeVars(id):
    """Returns a dict of id's config. Creates a blank config if none found"""
    if not os.path.exists(f"Data/{id}/Runtime.json"):
        genRuntimeVars(id)
    with open(f"Data/{id}/Runtime.json", "r") as f:
        return json.load(f)


def clearRuntimeVars(id):
    """Replaces runtime vars with default"""
    data = defaultRuntimeVars
    data["id"] = id
    if not os.path.exists(f"Data/{id}/Runtime.json"):
        genRuntimeVars(id)
    with open(f"Data/{id}/Runtime.json", "w") as f:
        json.dump(data, f)


def getSDPayload(type):
    with open(f"Payloads.json", "r") as f:
        return json.load(f)[type]


def getUserData(userID):
    with open(f"Data/{userID}/User.json", "r", encoding="utf-8") as f:
        return json.load(f)


def setUserData(userID, new):
    if os.path.exists(f"Data/{userID}/User.json"):
        with open(f"Data/{userID}/User.json", "w+", encoding="utf-8") as f:
            json.dump(new, f, indent=4)


# vars = {"user": "Clay", "char": "Loona"}
# characterJson = getCharacterPrompt(ADMIN_ID, "Loona")
# print(varInsert(processJsonPrompt(characterJson), vars))


if __name__ == "__main__":
    main()
