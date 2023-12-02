import openai
import os
import json
import discord
from discord.ext import commands
import shutil

openai.api_key = ""
discord_bot_token = ''
triggers = ["hey gpt"] #This is how to activate the bot. The bot will respond when it detects this word in any message sent on a server.

#This is a simple script to converse with OpenAI's GPT models. It tries to keep persistence between chats by creating a file to store logs of the past conversations, here known as neuralcloud_discord.ncb. 
#Model responses are also written to a log.log for further reference.
#This script uses the chat model, or currently the gpt-3.5 model that is similar to ChatGPT.

#################
### Variables ###

#model is the used OpenAI model. Check their website for different model names.
#https://platform.openai.com/docs/models/overview
model="gpt-3.5=turbo"

#the prompt is what the model will read for to create the response.
#Do not include the initial human prompt, just the description of what the model's pesonality should be like.
base_prompt="""The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly."""

#Limit for how many pieces of dialogue the model should remember before summarizing the previous conversations back into the prompt.
#This is used as a way to prolong how much the model can talk to us before hitting the overall token limit.
limit_length=50

#################
#################

#First, a function to save the memory variable to the ncb. I will use this a lot, so it works best as a function.
def save_ncb():
  with open('neuralcloud_discord.ncb', 'w') as save:
     save.write(json.dumps(memory)) 

#Initialize my custom memory file. Basically, a text file to log everything we've written and then reuse it as the prompt for future prompts. 
#First we check if there already exists a neural cloud file. If not, then we create the ncb file and wrtie the prompt to it.
#Its Like waking up their neural cloud for the first time. Otherwise, its just restoring their neural clouds.
memory=[] #unlike the gpt3 script, we use a variable to store memory here. 
ncb = './neuralcloud_discord.ncb'
check = os.path.isfile(ncb)
if check:
  with open('neuralcloud_discord.ncb') as read:
    output = read.read()
  formatted_list = json.loads(output)
  memory = formatted_list #These steps allow the model to have past dialogues loaded as a python list
else:
  memory.append({"role": "system", "content": f"{base_prompt}"}, ) #creating the file with the system prompt
  memory.append({"role": "user", "content": "Hello."}, )
  save_ncb() #So the model's first words are a greeting to the user.

#################
### Functions ###

#Function for the api request so that I don't have to copy paste this over and over again.
def api_request(prompt):
  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=prompt
  )
  api_request.response = response['choices'][0]['message']['content'].strip() 
  memory.append({"role": "assistant", "content": f"{api_request.response}"}, ) #write to the memory variable
  save_ncb() #save memory to ncb after generation of response
  log = open("log_discord.log", "a")
  log.write("\n" + "Assistant: " + api_request.response) #Write to log
  log.close()

#Function to determine how to compress the ncb
def cleaner():
  global memory
  if len(memory) >= limit_length:
    # GOALS:
    # Make the summaries additive rather than replacing them altogether. Consider modifying the log file by adding in the previous summary as well.
    # IMPLEMENTED as putting in the new_prompt into the log before the user / assistant dialogue 
    # CHECK TO SEE IF IT WORKS
    
    ##Summarizer
    print("Cleaning up neural cloud. Please wait...") #print so that user can see what is going on
    with open('log_discord.log') as read: #the log will show both user and assistant dialog. This makes it perfect for the summarizer.
      output = read.read()
    query="Only summarize the following conversation into one line from the perspective of the assistant. Do not explain." + '"' + output + '"' #this is the prompt for summary sent to the api
    summary=[] #again, the api requires a list rather than text
    summary.append({"role": "system", "content": f"{query}"}, )
    summarize = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=summary
    )
    summarize.response = summarize['choices'][0]['message']['content'].strip() 
    new_prompt=base_prompt + "\n" + "A summary of their previous conversation is as follows: " + summarize.response #now we need to replace the old memory variable with the new prompt
    memory=[] #blank out the memory variable
    memory.append({"role": "system", "content": f"{new_prompt}"}, ) #add in the new prompt (base_prompt + summary) to the memory variable

    ## File manipulation First we remove both backup files, should they exist
    if os.path.exists("neuralcloud_discord.ncb.bk"):
      os.remove("neuralcloud_discord.ncb.bk")
    else:
      pass
    if os.path.exists("log_discord.log.bk"):
      os.remove("log_discord.log.bk") 
    else:
      pass
    original_ncb = r'neuralcloud_discord.ncb'
    backup_ncb = r'neuralcloud_discord.ncb.bk' #makes the ncb backup
    shutil.copyfile(original_ncb, backup_ncb)
    original_log = r'log_discord.log'
    backup_log = r'log_discord.log.bk' #makes the log backup
    shutil.copyfile(original_log, backup_log)
    os.remove("neuralcloud_discord.ncb")
    os.remove("log_discord.log") #remove both original files
    save_ncb() #make a new ncb file, with the new_prompt as the system content
    log = open("log_discord.log", "a")
    log.write("A summary of the previous conversation is as follows: " + summzarize.response) #Write to log the summary part as well, just so that we don't lose bits of the memory from pre-clean.
    log.close()
  else:
    pass

#################################################################################
# This is the discord bot portion.

intents = discord.Intents().all()
client = commands.Bot(command_prefix=',', intents=intents)

@client.event
async def on_ready():
    print('online')
    print(memory)

@client.event 
async def on_message(message):
    if message.author == client.user:
        return
    if message.author.bot: 
        return
    for i in range(len(triggers)):
        if triggers[i].lower() in message.content.lower():
            cleaner()
            memory.append({"role": "user", "content": message.content}, )
            save_ncb()
            api_request(memory)
            print(api_request.response)
            await message.channel.send(api_request.response)

client.run(discord_bot_token)
