import openai
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from pymongo.server_api import ServerApi
import random

load_dotenv()
connection_string = os.getenv("MongoDB_CONNECTION_STRING")
AI_key = os.getenv("OpenAI_API_KEY")
client = MongoClient(connection_string, server_api=ServerApi('1'))
collection = client['Disinformation']["Labeled Data"]


topics = [
    'Flat Earth',
    'Covid-19',
    'Aliens',
    'Politics',
    'Sports',
    'Entertainment',
    'Technology',
    'Science',
    'A Celebrity',
    'A Conspiracy Theory',
    'Medicine',
    'Food',
    'Healthcare System',
    'Climate',
    'Education',
    'Privacy',
    'Animals',
    'Water sources',
    'Countries'
] 


labels = [
    'Disinformation',
    'Misinformation',
    'Satire/Joke',
    'None',
    'Unknown'
]

info = [
    "You will be tweeting Disinformation. You know that the information you are tweeting about is false, but you wish to spread some kind of agenda.",
    "You will be tweeting Misinformation. You don't know that the information that you are tweeting is false, you are just misinformed. You are not trying to spread any kind of agenda.",
    "You will be tweeting Satire or Jokes. The information you are tweeting may or may not be true, but the purpose is to make people laugh and not have people take it seriously.",
    "You will be tweeting normally. The things you tweet are either objectively true, or contain no objective information and are a random opinion or statement that is neither misinformation or disinformation.",
    "You will be tweeting randomly. Your tweets could be true, misinformation, disinformation, or satire, but it will be either hard to tell or be completely random and unrelated."
]

def main():
    print("Generating data...")
    while(True):
        openai.api_key = AI_key

        int1 = random.randint(0, len(topics)-1)
        int2 = random.randint(0, len(labels)-1)

        topic = topics[int1]
        label = labels[int2]
        pinfo = info[int2]
        
        
        prompt = f"Generate a theoretical tweet about {topic}."
        
        print("prompt: \n", prompt)
        

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=[
                {"role": "system", "content": f"You are a twitter user. You can either tweet a standalone tweet or reply to another user. {pinfo}"},
                {"role": "user", "content": f"Generate a tweet about {topic}."}
            ]
        )
        
        print("\nresponse: \n", response["choices"][0]["message"]["content"])
        
        
        answer = input(f"\nConfirm {label} for db (Y/N or exit)- ")
        
        if(answer.lower() == "y"):
            document = {"text": response, "label": label}
            collection.insert_one(document)
            print("Sending to db...\n")
        
        elif(answer == "exit"):
            exit()
            
        else:
            print("Skipping...")

if __name__ == "__main__":
    main()