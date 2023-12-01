import torch
import gradio as gr
from torch import nn 
import cv2
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from time import sleep

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import tiktoken
import openai
import os
openai.api_type = "azure"
openai.api_version = "2023-05-15"
# Your Azure OpenAI resource's endpoint value .
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_key = os.getenv("AZURE_OPENAI_KEY")

system_message = {"role": "system", "content": "You are a helpful assistant."}
# role: system  -> sets the behaviour of the assistant, enables dev to frame the conversation without being part of the convo (whisper in his ear before going on stage)
# role: user -> you 
max_response_tokens = 250
token_limit = 4096
conversation = []
conversation.append(system_message)

AQLdetails = """
- depending on the amount of bad apples defined as x we decide the acceptable quality limit class (AQL) with the following logic:
for x = 'Blotch Apple'+'Rot Apple' + 'Scab Apple':
    if x=0 then AQL Class I, suitable for supermarket and export
    if 1<x=<8 then AQL CLass II, suitable for making applesauce
    if 8<x=<15 then AQL Class III, suitable for making syrup
    if x>15 then AQL Class IV, rejected apples
<<<<<<< HEAD
- Follow these steps to determine the AQL class:
    1. from the dictionary calculate the total amount of apples by adding the amount of each type
    2. from the dictionary get the amount of type 'Normal Apples'
    3. calculate the amount of bad apples by substracting the type 'Normal Apples' from the total amount of apples
    4. based on the amount of bad apple, also known as x, determine the AQL classification
=======
- always check before responding if the calculated x falls in the correct AQL Class but don't mention this check
>>>>>>> 75c7e47adfca08882e9594ddc9e56b22a8311719
- a good way to provide insight is to provide a table consisting of two columns. In the first column put the counts of each type of apple in descending order and in the second column put the percentage of occurance which is the count of the type of apple divided by the total amount of apples. Always check if the total percentages add up to 1 but don't mention it. Use an ASCII art bar graph to make the table pretty. 
"""

ChatBotGoal = f"Your task is to provide information and insights on the results of sampling a batch of apples based on the information provided in the AQL knowledge delimited by triple backticks and the model output provided as a dictionary. AQL knowledge: ```{AQLdetails}```. Keep the response short and concise no longer then 3 sentences"
conversation.append({"role": "system", "content": ChatBotGoal})

# Loading the classification model
modelresnet = torch.load('apple_resnet_classifier.pt',  map_location=torch.device('cpu'))
modelresnet.to(device)
modelresnet.eval()

# print('Enter image location:')
# image_url = input()
# img = Image.open(image_url)

# Make program known to user
print('''
    ___                __        ________                _ _____          
   /   |  ____  ____  / /__     / ____/ /___ ___________(_) __(_)__  _____
  / /| | / __ \/ __ \/ / _ \   / /   / / __ `/ ___/ ___/ / /_/ / _ \/ ___/
 / ___ |/ /_/ / /_/ / /  __/  / /___/ / /_/ (__  |__  ) / __/ /  __/ /    
/_/  |_/ .___/ .___/_/\___/   \____/_/\__,_/____/____/_/_/ /_/\___/_/     
      /_/   /_/                                                           
      ''')
# User input of sample location
print('Enter folder location:')
folder_url = input()
#folder_url = r"D:\apple_50sample"

transform_img_normal = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])


def predict(folder_path):
    class_labels = ['Blotch Apple', 'Normal Apple', 'Rot Apple', 'Scab Apple']
    class_counts = [0,0,0,0]

    dataset = ImageFolder(folder_path, transform=transform_img_normal)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=False)
    with torch.no_grad():
        for data in dataset_loader:
            inputs, labels = data

            #weights in the loaded model are cuda casted 
            #cast the inputs also to cuda to make it work 
            inputs = inputs.to(device)

            out = modelresnet(inputs).to(device)
            _, predicted = torch.max(out.data, 1)
            for p in tqdm(predicted):
                if p.item() == 0:
                    class_counts[0] += 1
                elif p.item() == 1:
                    class_counts[1] += 1
                elif p.item() == 2:
                    class_counts[2] += 1
                else:
                    class_counts[3] += 1
                sleep(0.1)
        
        label_counts_dict = {label: count for label, count in zip(class_labels, class_counts)}

        return label_counts_dict


# def predict(image):
#     img = image.resize((224, 224))
#     img = ToTensor()(img).unsqueeze(0).to(device)
#     with torch.no_grad():
#         out = modelresnet(img)
#         _, predicted = torch.max(out.data, 1)
#         probabilities = torch.nn.functional.softmax(out, dim=1)[0]
#         class_labels = ['Bad Apple', 'Normal Apple',
#                         'Rot Apple', 'Scab Apple']
#         values, indices = torch.topk(probabilities, 4)
#         confidences = {class_labels[i]: v.item() for i, v in zip(indices, values)}
#         print(confidences)
#         return confidences

def append_data(table):
    prompt = f"here are the results of the model in a dictionary {table}"
    table_input = {"role": "system", "content": prompt}
<<<<<<< HEAD
    #print(table_input)
=======
    print(table_input)
>>>>>>> 75c7e47adfca08882e9594ddc9e56b22a8311719
    conversation.append(table_input)
    print('Sample batch has been processed, you can now question the data in natural language:')

#x = predict(img)
x = predict(folder_url)
append_data(x)

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for message in messages:
        # every message follows <im_start>{role/name}\n{content}<im_end>\n
        num_tokens += 4
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens


while (True):
    print('Question(enter q to exit):')

    user_input = input("")

    if (user_input == 'q'):
        break

    conversation.append({"role": "user", "content": user_input})
    conv_history_tokens = num_tokens_from_messages(conversation)

    while (conv_history_tokens+max_response_tokens >= token_limit):
        del conversation[1]
        conv_history_tokens = num_tokens_from_messages(conversation)

    response = openai.ChatCompletion.create(
        # The deployment name you chose when you deployed the ChatGPT or GPT-4 model.
        engine="MyChatGPT35Turbo",
        messages=conversation,
        temperature=.7,
        max_tokens=max_response_tokens,
    )

    conversation.append(
        {"role": "assistant", "content": response['choices'][0]['message']['content']})
    print("\n" + response['choices'][0]['message']['content'] + "\n")
