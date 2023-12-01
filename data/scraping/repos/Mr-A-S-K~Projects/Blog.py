from dotenv import dotenv_values
import openai as AI
config = dotenv_values(".env")
AI.api_key = config["API_KEY"]

def Blog(Topic):
    response = AI.Completion.create(
        model="text-davinci-002",
        prompt = "Write a paragraph" + Topic,
        max_tokens = 400,
        temperature = 0.3
    )
    Store = response.choices[0].text
    return Store

Topic = input("What do you want to write about? ")
print(Blog(Topic))

Keep_writing = True
while Keep_writing:
    Continue = input("Do you want to keep writing? yes/no ")
    if Continue.lower() == "yes":
        print(Blog(Topic))
    else:
        Keep_writing = False
        print("Thank you for using Blog.py")
    