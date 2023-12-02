import openai
import os
import json
import shutil

openai.api_key = ""

#This is a simple script to converse with OpenAI's GPT models. It tries to keep persistence between chats by creating a file to store logs of the past conversations, here known as neuralcloudv3.ncb. 
#Model responses are also written to a log.log for further reference.
#This script uses the chat model, or currently the gpt-3.5 model that is similar to ChatGPT.

#counter variable that determines whether to begin with the model or the user. Do not change.
counter = 0

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
  with open('neuralcloudv3.ncb', 'w') as save:
     save.write(json.dumps(memory)) 

#Initialize my custom memory file. Basically, a text file to log everything we've written and then reuse it as the prompt for future prompts. 
#First we check if there already exists a neural cloud file. If not, then we create the ncb file and wrtie the prompt to it.
#Its Like waking up their neural cloud for the first time. Otherwise, its just restoring their neural clouds.
memory=[] #unlike the gpt3 script, we use a variable to store memory here. 
ncb = './neuralcloudv3.ncb'
check = os.path.isfile(ncb)
if check:
  with open('neuralcloudv3.ncb') as read:
    output = read.read()
  formatted_list = json.loads(output)
  memory = formatted_list #These steps allow the model to have past dialogues loaded as a python list
else:
  memory.append({"role": "system", "content": f"{base_prompt}"}, ) #creating the file with the system prompt
  memory.append({"role": "user", "content": "Hello."}, )
  save_ncb() #So the model's first words are a greeting to the user.
  log = open("logv3.log", "a")
  log.write("---split---") #Write a split character to log
  log.close()
  counter = 1 #now the model goes first.

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
  log = open("logv3.log", "a")
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
    with open('logv3.log') as read: #the log will show both user and assistant dialog. This makes it perfect for the summarizer.
      output = read.read()
      output0 = output.split("---split---")[0]
      output1 = output.split("---split---")[1]
    query="Only summarize the following conversation into one line from the perspective of the assistant. Do not explain." + '"' + output1 + '"' #this is the prompt for summary sent to the api
    summary=[] #again, the api requires a list rather than text
    summary.append({"role": "system", "content": f"{query}"}, )
    summarize = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=summary
    )
    summarize.response = summarize['choices'][0]['message']['content'].strip() 
    new_prompt=base_prompt + "\n" + "A summary of their previous conversation is as follows: " + output0 + summarize.response #now we need to replace the old memory variable with the new prompt
    memory=[] #blank out the memory variable
    memory.append({"role": "system", "content": f"{new_prompt}"}, ) #add in the new prompt (base_prompt + summary) to the memory variable

    ## File manipulation First we remove both backup files, should they exist
    if os.path.exists("neuralcloudv3.ncb.bk"):
      os.remove("neuralcloudv3.ncb.bk")
    else:
      pass
    if os.path.exists("logv3.log.bk"):
      os.remove("logv3.log.bk") 
    else:
      pass
    original_ncb = r'neuralcloudv3.ncb'
    backup_ncb = r'neuralcloudv3.ncb.bk' #makes the ncb backup
    shutil.copyfile(original_ncb, backup_ncb)
    original_log = r'logv3.log'
    backup_log = r'logv3.log.bk' #makes the log backup
    shutil.copyfile(original_log, backup_log)
    os.remove("neuralcloudv3.ncb")
    os.remove("logv3.log") #remove both original files
    save_ncb() #make a new ncb file, with the new_prompt as the system content
    log = open("logv3.log", "a")
    log.write(output0 + summarize.response + "\n" + "---split---") #Write to log the summary part as well, just so that we don't lose bits of the memory from pre-clean.
    log.close()
  else:
    pass

#Main function to regulate a question-answer type interaction between user and the model. First load in the past prompts, then move on.
def main():
  while True:
    global counter
    if counter == 1:
      api_request(memory)
      print(api_request.response)
      save_ncb()
    else:
      pass
    #Then have the user interact with the model.
    #Function to ask user for input
    counter = 1
    cleaner()
    user_input = input("[Enter your input]: ")
    memory.append({"role": "user", "content": f"{user_input}"}, )
    log = open("logv3.log", "a")
    log.write("\n" + "User: " + user_input)
    log.close()


if __name__ == "__main__":
    main()