from PIL import Image
import requests
import openai
import os 
import openai
import glob
import pandas as pd
import random
import csv

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

openai.api_key = "--------"

image_list = []

df1 = pd.read_csv('../data_provided/objects_combined.csv')
col_list1 = df1.object_name.values.tolist()


### IMPLEMENT ENSEMBLING FOR LATER ITERATIONS ###

for filename in glob.glob('images/*.png'):
 

    image=Image.open(filename)

    #Remove & .png#
    sub = filename[0:filename.find('/')+1]
    #png = filename[len(filename)-4:len(filename)]
    desc = filename.replace(sub,'')

    f = open('output/'+desc+'_objects_output.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(
        ["Correct Item", "Correct Item Value", "RH #1", "RH #1 Value", "RH #2", "RH #2 Value", "RH #3", "RH #3 Value", "RH #4", "RH #4 Value", "Result"]
    )
    
    desc = desc[:-4]
    desc = desc[desc.find(' - ')+3:len(desc)]
    print(desc)
    #-------#
    
    #Details#
    lists = []
    #-------#

    response = openai.Completion.create(
        model="text-davinci-002",
        prompt="List the one word objects in the following sentence ("+desc+") seperated by commas:",
        temperature=0,
        max_tokens=256,
    )
    objects = response.choices[0].text.strip().split(", ")

    for i in objects:
        if (i.find('a ') == 0): 
            i = i.replace('a ','', 1)
        if (i.find('A ') == 0): 
            i = i.replace('A ','', 1)
        if (i.find('an ') == 0): 
            i = i.replace('an ','', 1)
        if (i.find('An ') == 0): 
            i = i.replace('An ','', 1)
       
        ### MAKE SURE OBJECTS ARNT IN IMAGE FOR RED HERRING ###
        objectsNew = random.sample(col_list1, 4)
        objectsNew.insert(0, i)

        lists.append(objectsNew)

    for i in lists:
        inputs = processor(text=i, images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
        print("----------------------")    
    
        result = True
        
        row = []

        for z in range(len(i)):
            print(i[z] + ": " + str(probs.detach().numpy()[0][z]))
            row.append(i[z])
            row.append(str(probs.detach().numpy()[0][z]))

            if (float(probs.detach().numpy()[0][0]) < float(probs.detach().numpy()[0][z])):
                result = False
        
        if (result):
            print(" ## " + i[0] + ": " + "PASS!")
            row.append("PASS!")
        else: 
            print(" ## " + i[0] + ": " + "FAIL!")
            row.append("FAIL!")
        writer.writerow(row)
        
    writer.writerow(
        ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
    )
    f.close()
    print("----------------------")    


