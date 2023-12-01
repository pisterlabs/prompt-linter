#creates a new assistant
import config
import openai
import os
import time
import shutil

openai.api_key = config.openaiapikey

print("Enter the assistants Name")
gptname = input().lower()

print("Enter the instructions for the Assistant. How they should behave, speak etc.")
gptinstruction= input()

print("In which language will you speak to this assistant? e.g. en-US")
language = input()

print("Enter the ElevenLabs voice name that the assistant should use")
voiceid = input()

print("Do you want to use the gpt-4-turbo preview or gpt-3.5-turbo? Enter 4 or 3. ")
modelnumber = input()
if modelnumber == "3" :
    gptmodel = "gpt-3.5-turbo-1106"


elif modelnumber == "4": 
    gptmodel = "gpt-4-1106-preview"


else:
    print("Neither 3 or 4")
    time.sleep(2)
    quit()

assistant = openai.beta.assistants.create(
    name=gptname,
    instructions=gptinstruction,
    model=gptmodel
)

configfile = gptname + "_config" + ".py"

assistantfile = gptname + ".py"
src_dir = os.getcwd()
dest_dir = src_dir +  "\\" + assistantfile
shutil.copy('undefassistant.py', dest_dir)

#append assistant specifics to config file
with open(configfile, "w") as file:
    l1 = "\nassistantid = "
    l2 = '"'
    l3 = assistant.id
    l4 = '"\n'
    l5 = "voiceid = "
    l6 = '"'
    l7 = voiceid
    l8 = '"\n'
    l9 = "language = "
    l10 = '"'
    l11 = language
    l12 = '"'
    file.writelines([l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12])
    file.close

print(f"The Assistant has been created, you can now create a Thread with it by executing {assistantfile}")

