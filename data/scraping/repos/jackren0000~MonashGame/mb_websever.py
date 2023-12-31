# -*- coding: utf-8 -*-
"""MB_webSever

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1A4YzjeqqKXJuVEOWX2h1S40LwtTM_xaw

**Mount to Google Drive**
"""

import openai
openai.api_key = 'sk-kHl0AvuILHBMLSX4nd9lT3BlbkFJj7VdUTfpzg8GtBmPRQO5q'

story_so_far = "you are standing in front of a building."

def generate_next_step(action):
    global story_so_far
    story_so_far += "\n" + action
    
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt = f'In this immersive, text-based adventure set within the confines of Monash University after zombie apocalypse, you\'re the protagonist crafting the next move. Each action should logically follow from the previous events and should not contradict any established facts. Think outside the box and generate a creative, unexpected next move. Consider the implications of each decision, the potential reactions of other characters, and the overall narrative arc. Be creative, avoid repetition, and keep your narrative advancement succinct. Here\'s the story so far: {story_so_far}. What\'s your next move?.\n\n',
        max_tokens=500
    )
    story_so_far += "\n" + response.choices[0].text.strip()

    return response.choices[0].text.strip()







# flask is a Python library that lets you develop web servers.
from flask import Flask, request, jsonify, send_from_directory
import torch
from PIL import Image
from torchvision import transforms

import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
  def __init__(self, num_classes = 3):
    super().__init__()
    # First convolutional layer
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.pool_1 = nn.MaxPool2d(2, 2)

    # Second convolutional layer
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.pool_2 = nn.MaxPool2d(2, 2)

    # Fully connected layer
    self.fc1 = nn.Linear(53 * 53 * 16, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 3)

  def forward(self, x):
    # Pass the first convolutional layer
    x = self.pool_1(F.relu(self.conv1(x)))

    # Pass the second convolutional layer
    x = self.pool_2(F.relu(self.conv2(x)))

    # Flatten x into one dimension
    x = torch.flatten(x, 1)

    # Pass the fully connected layer
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)

    return x

  def predict(self, x):
    predictions = self(x)

    # Get the predicted classes
    _, predicted_classes = torch.max(predictions, dim=1)

    return predicted_classes, x
      
# Instantiate your model and load its weights
model = CNN()
model.load_state_dict(torch.load('CNN_model.pth'))
model.eval()

building_names = ['Monash Hargrave Andrew Library', 'Monash Menzies Building', 'Monash Sir Louis Matheson Library']

last_prediction = None
last_building_name = None

app = Flask(__name__)

# If there is a GET request to the root of the router, execute index() function.
@app.route('/', methods=['GET'])
def index():
    return send_from_directory('static', 'index.html')

# If there is a POST request to the predict endpoint of the router, execute predict() function
@app.route('/predict', methods=['POST'])
def predict():
  global last_prediction
  global last_building_name

  file= request.files.get('file', None)
  action = request.form.get('action', None)

  # Initialize prediction to None
  prediction = None

  if file is not None:
    image = Image.open(file.stream)
    transform = transforms.Compose(
      [transforms.Resize((224, 224)),
       transforms.ToTensor()])
    image = transform(image)
    image = image.unsqueeze(0)
    output = model(image)
    _, predicted = torch.max(output, 1)
    prediction = predicted.item()
    last_prediction = prediction 
    building_name = building_names[prediction]
    last_building_name = building_name
    action = f"I decided to enter the {building_name}."
  else:
      prediction = last_prediction
      building_name = last_building_name
    
  # Generate story part using action
  story_part = generate_next_step(action)
    
  return jsonify({'prediction': prediction, 'story': story_part, 'building_name' : building_name})


if __name__ == '__main__':
  app.run(debug = True)
