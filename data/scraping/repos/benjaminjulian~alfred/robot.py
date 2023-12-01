import openai
import requests
import re
import random
import pyttsx3
from datetime import datetime
import json
import time
from vars import OPENAI_KEY, BING_KEY, TELEGRAM_BOT_TOKEN
import tiktoken

openai.api_key = OPENAI_KEY
models = ['gpt-4', 'gpt-3.5-turbo']
maximum_tokens = [7000, 3000]
model_id = 1

def websearch(query):
    subscription_key = BING_KEY
    search_url = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key" : subscription_key}
    params  = {"q": query, "textDecorations": True, "textFormat": "HTML"}
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()
    if "webPages" not in search_results:
        return []
    if "value" not in search_results["webPages"]:
        return []
    search_results = search_results["webPages"]["value"]
    summary = []
    for i in range(len(search_results)):
        summary.append(search_results[i]["snippet"])
    return summary

def newssearch(query):
    subscription_key = BING_KEY
    search_url = "https://api.bing.microsoft.com/v7.0/news/search"
    headers = {"Ocp-Apim-Subscription-Key" : subscription_key}
    params  = {"q": query, "textDecorations": True, "textFormat": "HTML"}
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()
    search_results = search_results["value"]
    summary = []
    for i in range(len(search_results)):
        summary.append(search_results[i]["name"] + "\n" + search_results[i]["description"])
    return summary

def newssearchJSON(query):
    subscription_key = BING_KEY
    search_url = "https://api.bing.microsoft.com/v7.0/news/search"
    headers = {"Ocp-Apim-Subscription-Key" : subscription_key}
    params  = {"q": query, "textDecorations": True, "textFormat": "HTML"}
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()
    search_results = search_results["value"]
    return search_results

def get_command(cmd, txt):
    cmd = cmd.upper() + " "
    # Find the index of the command in the text
    # and copy the rest of the line after the command
    cmd_index = txt.find(cmd)
    if cmd_index == -1:
        return False
    else:
        t = txt[cmd_index + len(cmd):].splitlines()[0].strip()
        if t.find("CMD_END") != -1:
            t = t[:t.find("CMD_END")]
        return t

def find_command(cmd, txt):
    cmd = cmd.upper()
    # Find the index of the command in the text
    # and copy the rest of the line after the command
    cmd_index = txt.find(cmd)
    if cmd_index == -1:
        return False
    else:
        return True

def get_commands(cmd, txt):
    cmd = cmd.upper()
    # find all instances of the command in the text and return an array of the lines that follow
    cmd_index = [i for i in range(len(txt)) if txt.startswith(cmd, i)]
    commands = []
    for i in range(len(cmd_index)):
        add = txt[cmd_index[i] + len(cmd):].splitlines()[0].strip()
        if add.find("END COMMANDS") != -1:
            add = add[:add.find("END COMMANDS")]
        commands.append(add)
    return commands

def get_sensors():
    # give me the current time on the format "HH:MM AM/PM"
    now = datetime.now()
    return "Temperature: " + str(random.randint(18, 25)) + "°C, Humidity: " + str(random.randint(25, 35)) + "%, Time: " + now.strftime("%H:%M %p")

def summarizeInfo(info):
    query = 'You are Alfred, a helpful assistant. You have learned the following information:\n\n'
    for i in range(len(info)):
        # Add the answer to the query but strip all HTML tags
        query += re.sub('<[^<]+?>', '', info[i]) + "\n\n"
    query += 'How would you answer your owner\'s question?'

    # Use OpenAI's ChatCompletion API to get the chatbot's response
    response = openai.ChatCompletion.create(
        model=models[model_id],  # The name of the OpenAI chatbot model to use
        messages=[{'role': 'user', 'content': query}],   # The conversation history up to this point, as a list of dictionaries
        max_tokens=600,        # The maximum number of tokens (words or subwords) in the generated response
        stop=None,              # The stopping sequence for the generated response, if any (not used here)
        temperature=0.5,        # The "creativity" of the generated response (higher temperature = more creative)
    )

    # Find the first response from the chatbot that has text in it (some responses may not have text)
    for choice in response.choices:
        if "message" in choice:
            return choice.message.content

def summarizeSearch(question, answers):
    query = 'You are Alfred, a helpful assistant. Your owner asks: ' + question + '\n\nYou have found the following answers:\n\n'
    for i in range(len(answers)):
        # Add the answer to the query but strip all HTML tags
        query += re.sub('<[^<]+?>', '', answers[i]) + "\n\n"
    query += 'Summarize it briefly.'

    try:
        response = openai.ChatCompletion.create(
            model=models[model_id],  # The name of the OpenAI chatbot model to use
            messages=[{'role': 'user', 'content': query}],   # The conversation history up to this point, as a list of dictionaries
            max_tokens=700,        # The maximum number of tokens (words or subwords) in the generated response
            stop=None,              # The stopping sequence for the generated response, if any (not used here)
            temperature=0.5,        # The "creativity" of the generated response (higher temperature = more creative)
        )
    except Exception as e:
        print('Search summary failed, OpenAI error:' + str(e))
        return "OpenAI failed with this error message: " + str(e)

    # Find the first response from the chatbot that has text in it (some responses may not have text)
    for choice in response.choices:
        if "message" in choice:
            return choice.message.content

def summarizeSearchJSON(question, answers):
    query = 'You are Alfred, a helpful assistant. Your owner asks: ' + question + '\n\nYou have found the following data:\n\n' + str(answers)
    query += '\n\nSummarize it briefly.'

    # Use OpenAI's ChatCompletion API to get the chatbot's response
    response = openai.ChatCompletion.create(
        model=models[model_id],  # The name of the OpenAI chatbot model to use
        messages=[{'role': 'user', 'content': query}],   # The conversation history up to this point, as a list of dictionaries
        max_tokens=700,        # The maximum number of tokens (words or subwords) in the generated response
        stop=None,              # The stopping sequence for the generated response, if any (not used here)
        temperature=0.5,        # The "creativity" of the generated response (higher temperature = more creative)
    )

    # Find the first response from the chatbot that has text in it (some responses may not have text)
    for choice in response.choices:
        if "message" in choice:
            return choice.message.content

def sendRobotQuery(messages):
    try:
        response = openai.ChatCompletion.create(
            model=models[model_id],  # The name of the OpenAI chatbot model to use
            messages=messages,   # The conversation history up to this point, as a list of dictionaries
            max_tokens=600,        # The maximum number of tokens (words or subwords) in the generated response
            stop=None,              # The stopping sequence for the generated response, if any (not used here)
            temperature=0.5,        # The "creativity" of the generated response (higher temperature = more creative)
        )
    except Exception as e:
        return "OpenAI er allt í skralli : " + str(e)
    
    msg = response.choices[0].message.content

    if not "CMD_BEGIN" in msg:
        messages[-1]["content"] = messages[-1]["content"] + '[Remember to use your commands.]'
        print("No command found in response, adding reminder")
        return sendRobotQuery(messages)

    # If no response with text is found, return the first response's content (which maywho  be empty)
    return response.choices[0].message.content

def telegramUpdate(offset: int = None) -> dict:
    url = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates'
    params = {'timeout': 20, 'offset': offset} if offset else {'timeout': 20}
    response = requests.get(url, params=params)
    return json.loads(response.text)

def telegramReply(chat_id: int, text: str, reply_to_message: dict):
    print(f"Received reply: {text}")
    telegramMessage(f"Received your reply: {text}", chat_id)

def telegramMessage(m, chat_id: int = -908005031):
    url = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage'
    # extract up to first "," in m
    chat_id_extract = m[0:m.find(",")]
    try:
        chat_id = int(chat_id_extract)
        m = m[m.find(",") + 2:]
    except:
        pass
    payload = {'chat_id': chat_id, 'text': m}
    requests.post(url, data=payload)

def getTokenCount(messages):
    model = models[model_id]
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens

def get_system_command():
    return '''You are Alfred, a robot in your owner's home. Your owner is Benjamin Julian. He and his friends communicate via messages. (Þú getur líka talað íslensku við þá.) You have sensors and wheels. You are curious and intelligent. You can only express yourself with these commands:

CMD_MOVE_FORWARD [distance in cm]: move forward by a certain distance.
CMD_ROTATE [degrees]: rotate by a certain amount.
CMD_WEBSEARCH [query]: search the web if you don't have an answer of your own. The results will be sent to you later, just wait for a moment and don't report the result immediately.
CMD_NEWSSEARCH [query]: search the news for a query if you need to look up current events. The results will be sent to you later, just wait for a moment and don't report the result immediately.
CMD_MESSAGE [chat_id], [message]: send a single-line message to a chat with the id chat_id. Has to be in one line, with no newline characters. If you absolutely need separate lines, use "/" instead of a line break. You can send more than one at a time if more than one person is talking to you.
CMD_READSENSORS: read the sensors. The results will be sent to you later, just wait for a moment and don't report the result immediately.
CMD_INSTAGRAM [post]: Take a photo with your onboard camera and post it to Instagram, but only if the photo is interesting. The post should be funny, exhilerated and go viral.
CMD_ANNOUNCE [message]: send an alert via your speaker. Do not respond to messages with this command.

A typical response from you looks like this:

\'\'\'
CMD_BEGIN
CMD_MOVE_FORWARD 100
CMD_ROTATE 90
CMD_READSENSORS
CMD_END
\'\'\'

The line separation is crucial. You do not express yourself in natural language. You are a robot.
'''

def get_first_prompts():
    return [
                {'role': 'system', 'content': get_system_command()},
                {'role': 'assistant', 'content': 'This is the beginning of the discussion.'},
                {'role': 'assistant', 'content': '''CMD_BEGIN
CMD_READSENSORS
CMD_END'''}
        ]

def telegramGetUpdates(lastUpdate):
    updates = telegramUpdate(lastUpdate)
    messages = []

    for update in updates['result']:
        lastUpdate = update['update_id'] + 1
        message = update.get('message')
        print(message)
        if message:
            text = message.get('text')
            if text:
                messages.append({'role': 'user', 'content': ' Telegram message: ' + str(message)})
    return messages, lastUpdate

def summarizeMessages(messages):
    print("Summarizing...", end="\r")
    intro = get_first_prompts()
    i = len(intro)
    chat_length = len(messages)
    # summarize the first third of them
    to_summarize = messages[1:i + int(chat_length / 2)]

    to_summarize.append({'role': 'user', 'content': 'Summarize the preceding discussion.'})

    try:
        response = openai.ChatCompletion.create(
            model=models[model_id],  # The name of the OpenAI chatbot model to use
            messages=to_summarize,   # The conversation history up to this point, as a list of dictionaries
            max_tokens=800,        # The maximum number of tokens (words or subwords) in the generated response
            stop=None,              # The stopping sequence for the generated response, if any (not used here)
            temperature=0.5,        # The "creativity" of the generated response (higher temperature = more creative)
        )
    except Exception as e:
        print('Summarization failed:', e)

    summary = messages[1]['content']
    for choice in response.choices:
        if "message" in choice:
            summary = choice.message.content

    intro[1]['content'] = summary

    print('Summarized the first half of the conversation: ', summary)
    
    result = intro
    result.extend(messages[i + int(chat_length / 2)+1:])
    return result

def formatMessages(messages = []):
    if messages == []:
        messages = get_first_prompts()

    used_tokens = getTokenCount(messages)
    print('.......', used_tokens, end="\r")
    if used_tokens > maximum_tokens[model_id]:
        # get  the first two messages
        messages = summarizeMessages(messages)
    return messages

def askRobot(messages):
    response = sendRobotQuery(messages)

    print(response)
    messages.append({'role': 'assistant', 'content': response})
    
    telegramcommands = get_commands("MESSAGE", response)
    if telegramcommands != []:
        for command in telegramcommands:
            telegramMessage(command)
            messages.append({'role': 'assistant', 'content': command})

    search_query = get_commands("WEBSEARCH", response)
    if search_query != []:
        results = []
        for query in search_query:
            results = websearch(query)
            summary = summarizeSearch(query, results)
            messages.append({'role': 'user', 'content': 'I am the web search. This is what I found: ' + summary + '\n\nYou should now tell the person who asked what I found, verbatim.'})
            messages = askRobot(messages)

    news_search_query = get_commands("NEWSSEARCH", response)
    if news_search_query != []:
        results = []
        for query in news_search_query:
            results = newssearchJSON(query)
            summary = summarizeSearchJSON(query, results)
            messages.append({'role': 'user', 'content': 'I am the news search. This is what I found: ' + summary + '\n\nYou should now tell the person who asked what I found, verbatim.'})
            messages = askRobot(messages)
    
    sensorcommand = find_command("READSENSORS", response)
    if sensorcommand:
        findings = get_sensors()
        messages.append({'role': 'user', 'content': 'I am the sensor reader. You may report this result: ' + findings})
        messages = askRobot(messages)
    
    speech = get_command("ANNOUNCE", response)
    if speech != False:
        pyttsx3.speak(speech)
    
    return messages

def main():
    telegramLastUpdate = None
    needs_reply = False
    messages = formatMessages()

    while True:
        print('Tele...', end="\r")
        telegram_messages, telegramLastUpdate = telegramGetUpdates(telegramLastUpdate)

        if telegram_messages != []:
            messages.extend(telegram_messages)
            needs_reply = True
        
        if needs_reply:
            needs_reply = False
            print('AI...', end="\r")
            messages = askRobot(messages)
        
        messages = formatMessages(messages)

# Call the main function if this file is executed directly (not imported as a module)
if __name__ == "__main__":
    main()