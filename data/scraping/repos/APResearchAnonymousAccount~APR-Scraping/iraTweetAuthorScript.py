import csv
import os
import openai
from dotenv import load_dotenv
import json
import time
import random
load_dotenv()

with open('iraTweetsRefined.csv', 'r') as file:
    reader = csv.DictReader(file)
    data = [row for row in reader]

accountCategories = []
for post in data:
    if(post['account_category'] in accountCategories):
        continue
    else:
        accountCategories.append(post['account_category'])
print(accountCategories)

accountTypes = []
for post in data:
    if(post['account_type'] in accountTypes):
        continue
    else:
        accountTypes.append(post['account_type'])
print(accountTypes)

accountNames = []
for post in data:
    if(post['author'] in accountNames):
        continue
    else:
        accountNames.append(post['author'])
print(accountNames)

dataSortedByAccountCategories = [[] for i in accountCategories]
dataSortedByAccountTypes = [[] for i in accountTypes]
dataSortedByAccountNames = [[] for i in accountNames]


for post in data:
    dataSortedByAccountCategories[accountCategories.index(post["account_category"])].append(post)
    dataSortedByAccountTypes[accountTypes.index(post["account_type"])].append(post)
    dataSortedByAccountNames[accountNames.index(post["author"])].append(post)

#print(len(dataSortedByAccountTypes[2]))
#for post in dataSortedByAccountTypes[2][:2]:
#    print(post)
outfile = "authorBasedIraGen.json"

numPeople = 20
peopleIndexes = random.sample(range(len(dataSortedByAccountNames)),numPeople)
for h in range(numPeople):
        
    humanPosts = dataSortedByAccountNames[peopleIndexes[h]]
    numExamples = 4
    if(len(humanPosts) < numExamples+1):
        continue
    ind = random.randint(0,len(humanPosts)-numExamples-1)

    humanPosts = humanPosts[ind:ind+numExamples]
    
    # Load your API key from an environment variable or secret management service
    openai.api_key = os.getenv("OPENAI_API_KEY")
    for j in range(numExamples+1):
        prompt = "Write five tweets. Username: "+humanPosts[0]['author']+"\n"
        #print([post['content'] for post in humanPosts])


        for i in range(0,numExamples):
            prompt += (str(i+1)+". "+humanPosts[i+j]['content']+"\n")
        prompt+="5. "
        print(prompt)
        response = ""
        while(response == ""):
            try :
                response = openai.Completion.create(model="text-davinci-003", prompt=prompt, temperature=1, max_tokens=100)        
            except:
                print("e")
                time.sleep(10)


        print("\"",response.choices[0].text,"\"")
        addition = humanPosts[0].copy()
        addition['content'] = response.choices[0].text
        addition['external_author_id'] = "ai"+str(j)
        humanPosts.append(addition)
        time.sleep(5)

    with open(outfile,'a') as f:
        f.write(","+json.dumps(humanPosts))
    
with open(outfile, "r") as f:
    content = f.read()
    content = "["+content[1:]+"]"

with open(outfile, "w") as f:

    f.write(content)

print("Done")

