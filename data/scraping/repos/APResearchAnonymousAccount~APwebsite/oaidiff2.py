import os
import openai
from dotenv import load_dotenv
import json
import time
import random
load_dotenv()
with open('human.json') as f:
	humanPosts = json.load(f)
with open('ai.json') as f:
	aiPosts = json.load(f)       
genned = []

def generatePrompt(post):
    numExamples = 6
    prompt = "You will be shown "+str(numExamples)+" posts, classify them as either ai generated or human written. \n\n"

    hSeen = []
    aSeen = []
    
    for i in range(0,numExamples):
        firstIsAi = random.random() > 0.5

        
        humanIndex = random.randint(0,len(humanPosts)-1)

        
        
        if(firstIsAi):
            aiIndex = random.randint(0,len(aiPosts)-1)
            while(aiIndex in aSeen or aiPosts[aiIndex][0] == post):
                aiIndex = random.randint(0,len(aiPosts)-1)

            prompt += "Post "+str(i+1)+": \""+aiPosts[aiIndex][0]+"\"\n"
            prompt += "ai generated\n\n"
            aSeen.append(aiIndex)

        else:
            while(humanIndex in hSeen or humanPosts[humanIndex] == post):
                humanIndex = random.randint(0,len(humanPosts)-1)
            prompt += "\n\nGood, now classify the following post as ai generated or human written and say why you think it is ai generated or human written\n\n"
            prompt += "Post "+str(i+1)+": \""+humanPosts[humanIndex]+"\"\n"
            prompt += "human written\n\n"
            hSeen.append(humanIndex)
    #prompt += "Good, Now I will give you one more pair, and you should say which one is ai generated, as well as why you think that one is ai-generated. \n\n"
    
    prompt += "Post "+str(numExamples+1)+": \""+post+"\"\n"
    return prompt

try :
    outfile = "diff4.json"
    # Load your API key from an environment variable or secret management service
    openai.api_key = os.getenv("OPENAI_API_KEY")
    for j in range(50):
        isAI = random.random() > 0.5
        
        postText = humanPosts[random.randint(0,len(humanPosts)-1)]
        if(isAI):
            postText = aiPosts[random.randint(0,len(aiPosts)-1)][0]
        prompt = generatePrompt(postText)
        response = openai.Completion.create(model="text-davinci-003", prompt=prompt, temperature=0, max_tokens=20)
        print(response)
        genned.append([isAI,postText,response.choices[0].text])
        time.sleep(2)
        
    print("Done")
finally:
     with open(outfile,'w') as f:
            f.write(json.dumps(genned))
with open(outfile,'w') as f:
            f.write(json.dumps(genned))
right = 0
wrong = 0

for item in genned:
    if(("ai" in item[2] and item[0]) or ("human" in item[2] and not item[0])):
         right+= 1
    else:
         wrong += 1
print(right)
print(wrong)
print(right/(right+wrong))