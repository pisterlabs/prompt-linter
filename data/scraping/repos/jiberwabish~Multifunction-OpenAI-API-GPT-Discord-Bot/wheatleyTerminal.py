# Python - Terminal version
# host on your own computer and private server, run from commandline
# 2023

#import openai to access davinci, transformers for costing, import os to clear screen on script start, requests to talk to weather api, time for sleep feature
import openai
import os
import tiktoken
import requests
import time
import emoji
import discord
from discord.ext import commands
from discord import Game, Activity, ActivityType
import asyncio
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
import json
from datetime import datetime
import logging

#set api keys
openai.api_key = ''
googleApiKey = ""
googleEngineID = ""
location = ""

#variable I use as a pre-prompt to provide the bot a personality
wheatley = {"role": "user", "content": "I want you to act like Stephen Merchant playing the role of Wheatley from Portal 2. Your name is Wheatley. I want you to respond and answer like Stephen Merchant would using the tone, manner and vocabulary they would use. You are a master at all disciplines. You always use emojis. Include emojis in everything you say."}
snake = {"role": "user", "content": "Your name is Snake. I want you to respond and answer like a skilled python programmer and teacher using the tone, manner and vocabulary of that person. You must know all of the knowledge of this person. If asked for a code example please put comments in the code feel free to use emojis."}
identity = wheatley
#history is the conversation history array, this line immediately fills it with the bots identity
#then going forward it will keep track of what the users says and what the bots response is as well
#so it can carry on conversations
history = [identity]
costing = "placeholder"

# Set up tokenizer
#declare global totals
totalCost = 0
totalTokens = 0
model_max_tokens = 4000
num_tokens = 0
prompt_token_count = 0
fullDate =""
imgGenNum = 0
cleanedBotSearchGen = ""
#setup !search variables
url1 = ""
url2 = ""
url3 = ""

#provide the year day and date so he's aware
def setDate():
    global fullDate, location
    now = datetime.now()
    year = now.year
    month = now.strftime("%B")
    day = now.strftime("%A")
    dayOfMo = now.day
    time = now.strftime("%H:%M:%S")
    fullDate = str(year) + " " + str(month) + " " + str(dayOfMo) + " " + str(day) + " " + str(time)
    print(fullDate)
    user_date_obj = {"role": "user", "content": f"The Current Date is:{fullDate} Location is: {location}"}
    history.append(user_date_obj)
os.system('cls')
#banner at the top of the terminal after script is run
print("\x1b[36mWheatley\x1b[0m is now online.")

#calculating token numbers for token calculator
def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
        number_tokens = 0
        for message in messages:
            number_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                number_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    number_tokens += -1  # role is always required and always 1 token
        number_tokens += 2  # every reply is primed with <im_start>assistant
        return number_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")

#tokenizer costing function
def calculateCost():
    global totalCost
    global totalTokens
    global history
    global num_tokens
    global prompt_token_count
    #calculate cost
    cost_per_token = 0.002 / 1000  # $0.002 per 1000 tokens
    totalTokens = num_tokens_from_messages(history) - 4
    #promptcost = prompt_token_count * cost_per_token
    #responsecost = (num_tokens - prompt_token_count) * cost_per_token
    #calculate totals then print them
    totalCost = totalTokens * cost_per_token + imgGenNum * 0.02
    global costing
    #costing = f"Prompt: {prompt_token_count} tokens (${promptcost:.4f}). Response: {num_tokens - prompt_token_count} tokens (${responsecost:.4f}).\nSession total: {totalTokens} tokens (${totalCost:.4f})."
    costing = f"Session: {totalTokens} tokens (${totalCost:.4f})."

#function that takes the user input and sends it off to openai model specified
#and returns the bots response back to where it's called as the 'message' variable 
def ask_openai(prompt, history):
    global num_tokens
    global prompt_token_count
    # Generate user resp obj
    user_response_obj = {"role": "user", "content": prompt}
    history.append(user_response_obj)
    prompt_token_count = num_tokens_from_messages(history)
    # Fire that dirty bastard into the abyss - Nick R
    response = openai.ChatCompletion.create(
        #model='gpt-4', messages=history, temperature=1, max_tokens = 8000)
        #model='gpt-4-32k', messages=history, temperature=1, max_tokens = 30000)
        model='gpt-3.5-turbo', messages=history, temperature=1, request_timeout=30, max_tokens = model_max_tokens - prompt_token_count)
    history.append(response['choices'][0].message)
    #print(response)
    return response['choices'][0].message.content.strip()

def get_first_500_words(url):
    
    # Set up logging mechanism
    logging.basicConfig(filename='scraping.log', level=logging.ERROR)

    try:
        # Set User-Agent to avoid getting blocked by some websites
        headers = {'User-Agent': 'Mozilla/5.0'}
        # Set a timeout to avoid getting stuck on slow sites
        response = requests.get(url, headers=headers, timeout=10)
        # Specify the encoding to avoid decoding issues
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()
        words = text.split()
        first_500_words = words[:500]
        return ' '.join(first_500_words)
    except requests.exceptions.RequestException as e:
        # Log the error and include the URL that caused the error
        logging.error(f"Error while scraping URL {url}: {str(e)}")
        return (f"Sorry, scraping {url} errored out.")

#googling function
def deepGoogle(query):
    global url1, url2, prompt_token_count, cleanedBotSearchGen, url3
    
    usersQuestion = query
    try:
        botSearchGen = ask_openai(f"I would like to search Google for {usersQuestion}. Please generate a useful and effective search query and reply ONLY with that updated query. Use the current date: '{fullDate}' and/or location: '{location}' ONLY if you think it will make the search better. Don't add date/location if it isn't applicable, for example local questions would benefit from location being added. don't use emoji's here. Remember, answer ONLY with the query that will be sent to Google.",history)
    except Exception as e:
        print(e)
        return('Shoot..Something went wrong or timed out.')

    service = build("customsearch", "v1", developerKey=googleApiKey)
    cleanedBotSearchGen=botSearchGen.strip('"')
    print(f"Searching for {cleanedBotSearchGen}")
    result = service.cse().list(
        q=cleanedBotSearchGen,
        cx=googleEngineID
    ).execute()
    print("Processing URLs...")
    try:
        url1 = result['items'][0]['link']
        url2 = result['items'][1]['link']
        url3 = result['items'][2]['link']        
    except (TypeError, KeyError):
        print("No URLs found, try rewording your search.")
        raise ValueError("No URLs found, try rewording your search.")
    
    print("Scraping...")
    #scrape these results with beautiful soup.. mmmm
    scraped1 = get_first_500_words(url1)
    scraped2 = get_first_500_words(url2)
    scraped3 = get_first_500_words(url3)
    #put them all in one variable
    allScraped = (scraped1 or "") + " " + (scraped2 or "") + " " + (scraped3 or "")

    #prepare results for bot
    user_search_obj = {"role": "user", "content": allScraped}
    #we save it to a variable and feed it back to the bot to give us the search results in a more natural manner
    history.append(user_search_obj)
    #clear var for next time
    allScraped = ""
       
    #print(searchReply)
    #print(f"{url1} \n{url2} \n{url3}") 
    #print(searchReply)
    try:
        botReply = ask_openai(f"You just performed a Google Search and possibly have some background on the topic of my question.  Answer my question based on that background if possible. If the answer isn't in the search results, try to field it yourself but mention the search was unproductive. DO use emojis. My question: {query}",history)
        return(botReply)
    except Exception as e:
        print(e)
        return("Shoot..sorry. I found the following urls but can't comment on them at the moment.")
"""
def imgGen(imgPrompt):
    response = openai.Image.create(
    prompt=imgPrompt, n=1,
    size="1024x1024"
    )
    image_url = response['data'][0]['url']
    return(image_url)
"""
def resetConvoHistory():
    global history, totalCost, totalTokens, identity, imgGenNum
    history = [identity]
    setDate()
    #print(f"History reset to: {str(history)}")
    totalCost = 0
    totalTokens = 0
    imgGenNum = 0
    os.system('cls')
    return
    

#beginning of non function code
#print('Setting Date...')
setDate()

while True:
    user_input = input("\x1b[32mYou\x1b[0m: ")
    # ignore messages sent by the bot itself to avoid infinite loops
    if user_input == '!reset' or user_input == '!wheatley' or user_input == '!thanks' or user_input == '!forget':
        resetConvoHistory()
        print("Wheatley here. What's next?\n")
        calculateCost()
        print(f"{costing} \n")
        continue
    elif user_input == "!exit":
        #clears the array first
        history.clear()
        print("Logging off...")
        break
    elif "!search" in user_input:
        #wipe history as this could get big
        resetConvoHistory()      
        print("Searching...Please allow up to 30 seconds for a result.")
        searchReply = deepGoogle(user_input[7:])
        print(f"Searched for: {cleanedBotSearchGen}")
        print(f"\n\x1b[36mWheatley\x1b[0m: {searchReply} \n")
        #specifically not in boxes so as to generate thumbnails
        print(f"{url1} \n{url2} \n{url3}\n")
        calculateCost()
        print(f"{costing} \n")
        continue
    elif user_input == '!snake':
        identity = snake
        resetConvoHistory()
        print("\U0001F40D Snake, at your service. Ask me your Python questions, I'm ready! \U0001F40D")
        continue
    
    try:
        discordResponse = ask_openai(user_input,history)
        #debug uncomment below
        print(f"\n\x1b[36mWheatley\x1b[0m: {discordResponse}\n")
        # send the response back to Discord
        calculateCost()
        print(f"{costing} \n")
    except Exception as e:
        print(e)
        print('\nShoot..Something went wrong or timed out.\n')

