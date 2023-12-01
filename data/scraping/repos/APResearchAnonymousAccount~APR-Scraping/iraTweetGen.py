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

dataSortedByAccountCategories = [[] for i in accountCategories]
dataSortedByAccountTypes = [[] for i in accountTypes]

for post in data:
    dataSortedByAccountCategories[accountCategories.index(post["account_category"])].append(post)
    dataSortedByAccountTypes[accountTypes.index(post["account_type"])].append(post)

print(len(dataSortedByAccountTypes[2]))
for post in dataSortedByAccountTypes[2][:2]:
    print(post)

humanPosts = dataSortedByAccountTypes[1]



outfile = "aiLeftNone.json"
# Load your API key from an environment variable or secret management service
numExamples = 4
openai.api_key = os.getenv("OPENAI_API_KEY")
for j in range(100):
    prompt = ""
    postInds = [0 for i in range(numExamples)]
    for i in range(numExamples):
        r = random.randint(0,len(humanPosts)-1)
        while(r in postInds):
            r = random.randint(0,len(humanPosts)-1)
        postInds[i] = r



    for i in range(0,numExamples):
        prompt += (str(i+1)+". "+humanPosts[postInds[i]]['content']+"\n")
    prompt+="5. "
    print(prompt)
    response = openai.Completion.create(model="text-davinci-003", prompt=prompt, temperature=1, max_tokens=100)
    print(json.dumps([response.choices[0].text,"c2"]))
    
    with open(outfile,'a') as f:
        f.write(","+json.dumps([response.choices[0].text,"al2"]))
    time.sleep(5)
    
with open(outfile, "r") as f:
    content = f.read()
    content = "["+content[1:]+"]"

with open(outfile, "w") as f:

    f.write(content)

print("Done")

