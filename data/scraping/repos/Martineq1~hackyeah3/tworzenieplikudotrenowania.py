import openai
import requests
import json
import os

API_KEY = "***" 
API_ENDPOINT = "https://api.openai.com/v1/chat/completions"
FolderPath = "F:\Git\hackyeah3\zawody"
FolderPath2 = "F:\Git\hackyeah3\plikitreningowe\\trening1.json"
datatraining = ""

Zawody = ["prawnik" , "informatyk" , "kierowca"]

for zawod in Zawody:
   folder = FolderPath + "\\" + zawod
   for filename in os.scandir(folder):
         with open (filename, "r") as f:
            data = json.loads(f.read())
            data = data.replace('\n', '')
            data = data.replace('\"' , '').replace("{" , "").replace("}","")
            data = data.rstrip('\r\n')
            trainingdata = """{"messages": [{"role": "system", "content": "Ten chatbot nazwany Munkiem pomaga mlodym osobom decydowac o wybraniu odpowiedniego zawodu "}, {"role": "user", "content": "Kim na podstawie tych danych pokazujacych cechy charakteru i umiejetności """ +  data + """ najlepiej aby dana osoba zostala"}, {"role": "assistant", "content": """ + '"' + zawod + '"' + """}]}"""
            datatraining = datatraining + "\n" + trainingdata
            f.close()
#   with open (FolderPath2, "w") as f:
            print(datatraining)
#           json.dump(datatraining, f) 

