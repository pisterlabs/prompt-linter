#Reference: https://www.youtube.com/watch?v=mozBidd58VQ&list=LL&index=9&t=918s, by Nicholas Renotte
import torch
from PIL import Image
from torch import nn, save, load #neural network class
from torch.optim import AdamW #implements AdamW algorithm
import torch.optim as optim
from torch.utils.data import DataLoader #to load buit-in data set from pytorch
import os
from torchvision import datasets, transforms
from torchvision.io import read_image
from torchvision.transforms import ToTensor #convert an image into tenser which is what will work in pytorch
import torchvision.transforms as T
import numpy as np


#Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

learningRate = 1e-3
weight_decay = 1e-5
batchSize = 32
#gradient = grad



#We need to downlaod our data set
train = datasets.FER2013(root='C:\\Users\\User\\Desktop\\EmotionData', split='train', transform= ToTensor()) #FER-2013 by Kaggle
dataset = DataLoader(train, 32) 
RESOURCES = {
        "train": ("train.csv", "3f0dfb3d3fd99c811a1299cb947e3131"),
        "test": ("test.csv", "b02c2298636a634e8c2faabbf3ea9a23"),
    }

#1,48,48 is the image shape of our dataset

#Image Classifier Neural Network
class ImageClassifier(nn.Module):
    def __init__(self, train): #this method is used to define the layers of the network, such as convolutional layers, linear layers, activation functions, etc
        super().__init__()
        self.model = nn.Sequential( #model is an attribute/"variable in the class"
            nn.MaxPool2d((2,2), stride = 2),
            nn.Conv2d(1,32,(3,3), stride = 1), #input=1=black and white, output=32, kernel size of 3*3 to capture finer details in the image
           #torch.nn.Conv2d(in_channels, out_channels, kernel_size)
           #pooling?To further reduce computation overhead
            nn.Conv2d(32,64,(3,3), stride = 1),
            nn.ReLU(),
            nn.Conv2d(64,128,(3,3), stride = 1), #what would be the maximum possible combinations of kernels?
            nn.ReLU(),
           #shaving 2 pixels of height and width of the image each time
          nn.Flatten(),
           #after flatten the layers into 1D, we will pass the input shape into linear layer...
          nn.Linear(128*18*18, 7) #in_features= the (number of channels after the "shaving process")*(size of the resulting image), (number of channels after the shaving process) is same as out_channels of the conv2D layer lying just above it.
           #7 labels in total
           #48*48-> 24*24-> 22*22-> 20*20 -> 18*18
        )


    def forward(self, x):
        return self.model(x)
print('hello')

#Loss, optimizer
if torch.cuda.is_available():
    model = ImageClassifier(train=True).to('cuda')
    print("Cuda is available")
else:
    model = ImageClassifier(train=True).to('cpu')
    print("Cuda is not available")

optimizer = optim.AdamW(model.parameters(), lr = learningRate, weight_decay = weight_decay)
lossFunc = nn.CrossEntropyLoss(weight = torch.tensor([2.0, 18.0, 2.0, 1.0, 1.5, 2.5, 1.5]).to('cuda'))
#0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
#Training
if __name__ == "__main__":
    with open('model_state.pt', 'rb') as file:
        model.load_state_dict(load(file))

    image = Image.open('Image56.jpg') #can put any kind of images

    # Resize the image to 48*48
    new_size = (48, 48)
    image = image.resize(new_size)
    transform = transforms.Compose([
    transforms.Grayscale(),       # Convert to grayscale
    ])

    # Apply the transformations
    image_transformed = transform(image)
    image_tensor = ToTensor()(image_transformed).unsqueeze(0).to('cuda')

    #0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
    print(torch.argmax(model(image_tensor))) 
    #torch.argmax() returns the indices of the maximum values of a tensor across a dimension
    

if torch.argmax(model(image_tensor)) == 0:
    print("Emotion: Angry")

elif torch.argmax(model(image_tensor)) == 1:
    print("Emotion: Disgust")

elif torch.argmax(model(image_tensor)) == 2:
    print("Emotion: Fear")

elif torch.argmax(model(image_tensor)) == 3:
    print("Emotion: Happy")

elif torch.argmax(model(image_tensor)) == 4:
    print("Emotion: Sad")

elif torch.argmax(model(image_tensor)) == 5:
    print("Emotion: Surprise")

elif torch.argmax(model(image_tensor)) == 6:
    print("Emotion: Neutral")
'''
 ##################################################################   
    for epoch in range(10): #10 epoches
        for batch in dataset:
            inputData,targetData = batch
            inputData, targetData = inputData.to('cuda'), targetData.to('cuda')
            yhat = model(inputData)
            loss = lossFunc(yhat, targetData)

            #Apply backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch:{epoch} loss is {loss.item()}") #calcalate the loss

    with open('model_state.pt', 'wb') as file:
        save(model.state_dict(), file)
#######################################################################
 with open('model_state.pt', 'rb') as file:
        model.load_state_dict(load(file))

    image = Image.open('Image52.jpg') #can put any kind of images

    # Resize the image to 48*48
    new_size = (48, 48)
    image = image.resize(new_size)
    transform = transforms.Compose([
    transforms.Grayscale(),       # Convert to grayscale
    ])

    # Apply the transformations
    image_transformed = transform(image)
    image_tensor = ToTensor()(image_transformed).unsqueeze(0).to('cuda')

    #0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
    print(torch.argmax(model(image_tensor))) 
    #torch.argmax() returns the indices of the maximum values of a tensor across a dimension
    

if torch.argmax(model(image_tensor)) == 0:
    print("Emotion: Angry")

elif torch.argmax(model(image_tensor)) == 1:
    print("Emotion: Disgust")

elif torch.argmax(model(image_tensor)) == 2:
    print("Emotion: Fear")

elif torch.argmax(model(image_tensor)) == 3:
    print("Emotion: Happy")

elif torch.argmax(model(image_tensor)) == 4:
    print("Emotion: Sad")

elif torch.argmax(model(image_tensor)) == 5:
    print("Emotion: Surprise")

elif torch.argmax(model(image_tensor)) == 6:
    print("Emotion: Neutral")

'''


import ChatGPT_API_EmotionalSupport
from ChatGPT_API_EmotionalSupport import get_completion
import openai
import os
from dotenv import load_dotenv
prompt = f"The person in front of me is feeling {emotion}, what can I do to help?"
print(prompt)
response = get_completion(prompt)

print(response)

