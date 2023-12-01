import openai
import cv2
from PIL import Image
import os
import requests
import random
#set openai api key and model
openai_credentials_file=open("SECRET.txt","r")
key=openai_credentials_file.readline().split(" ")[0]
openai_credentials_file.close()
openai_model="gpt-3.5-turbo"
openai.api_key = key
while True:
    random_letter=random.choice("abcdefghijklmnopqrstuvwxyz")
    prompt_text="Think of a cool randomized noun for a toddler to learn like firetruck or dinasaur, make sure that it is random and be sure to only respond with the word. I am going to give you the letter I want the word to start with, it is ",random_letter," Random noun for toddler: "
    completion=openai.ChatCompletion.create(
            model=openai_model,
            messages=[{"role": "user", "content": str(prompt_text)}],
    )
    response = openai.Image.create(
        prompt=completion.choices[0].message.content+", no words or letters, just a picture of a "+completion.choices[0].message.content,
        n=1,
        size="512x512"
    )
    image_url = response['data'][0]['url']
    url = image_url
    response = requests.get(url)
    with open('image.jpg', 'wb') as f:
        f.write(response.content)
    img = cv2.imread('image.jpg')
    cv2.imshow('image',img)
    cv2.waitKey(0)

    #get input
    input_text=input("What is this called? or enter q to quit: ")
    if input_text=="q":
        exit()
    else:
        if input_text.lower() in completion.choices[0].message.content.lower():
            print("Correct!, That is a ", completion.choices[0].message.content)
        elif completion.choices[0].message.content.lower() in input_text.lower():
            print("Correct!, That is a ", completion.choices[0].message.content)
        else:
            print("Incorrect, That is a ", completion.choices[0].message.content)
   
cv2.destroyAllWindows()