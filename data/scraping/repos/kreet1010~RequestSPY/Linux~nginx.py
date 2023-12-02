import openai	# for AI based request validation on how to 
import json	# for storing the checked request
import datetime	# for date based reporting
import time	# to keep a track of waiting time after rate limit excedes
import os

banner = """
    ____                             __  _____ ______  __
   / __ \___  ____ ___  _____  _____/ /_/ ___// __ \ \/ /
  / /_/ / _ \/ __ `/ / / / _ \/ ___/ __/\__ \/ /_/ /\  / 
 / _, _/  __/ /_/ / /_/ /  __(__  ) /_ ___/ / ____/ / /  
/_/ |_|\___/\__, /\__,_/\___/____/\__//____/_/     /_/   
              /_/                                      
              						Linux v:0.1"""
              					
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLACK = '\033[30m' 
BLUE = '\033[34m' 
CYAN = '\033[36m' 
RESET = '\033[0m'  
openAI_key = ""
LogFilePath = "/var/log/nginx/access.log"
todaysDate = datetime.date.today().strftime('%d-%m-%Y')
fileName = f'ReqCheck{todaysDate}.json'
FolderLocation = "/home/kali/Desktop/RequestSPY/ReportsJSON"
if not os.path.exists(FolderLocation):
	os.makedirs(FolderLocation)
fileAccessAddress = os.path.join(FolderLocation, fileName)

def generateTxt(prompt):
    openai.api_key = openAI_key
    response = openai.Completion.create(engine = "text-davinci-003", prompt=prompt, max_tokens = 300)
    return response.choices[0].text

print(YELLOW + banner + RESET)
print()
LogFile = open(LogFilePath, 'r')
count = 1
for i in LogFile:
	waitTime = 21 # Rate limit is 3 requests per minute. Thus 1 request every 20 seconds.
	print(GREEN + f"[+] Processing Request {count}...." + RESET)
	count +=1
	print(f"Forwading request: "+i)
	print(f">> Request forwarded to openai server.")
	SuspiciousnessTracker = generateTxt(i +' Verify the possible attacks for the following requests and return data only in the following json format {is Susspicious: boolean, attack name:String , ipAddress: String, userAgent:String, attacking Tool:String }, here ipAddress reffers to the ip of attacker from log and userAgent reffers to Browser. Remember to not generate any additional text other than the JSON file.')
	print(">> JSON response recieved from openai server")
	print(SuspiciousnessTracker)
	fileStore = open(fileAccessAddress, 'a')
	print(f">> Opened JSON file \'{fileAccessAddress}\'.") 
	json.dump(SuspiciousnessTracker, fileStore, indent = 4)
	fileStore.write("\n")
	print(f">> Data Written to \'{fileAccessAddress}\'")
	print(RED + f">> cooldown time {waitTime} seconds....\n\n" + RESET)
	time.sleep(waitTime)
	fileStore.close()  
LogFile.close()
print(CYAN + f"[+] Report generated is stored at: {fileAccessAddress}"+RESET)
# /home/kali/Desktop/RequestSPY/
