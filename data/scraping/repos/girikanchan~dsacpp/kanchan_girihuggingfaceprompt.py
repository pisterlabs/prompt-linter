from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests

#url = "file:///Users/ashokkumargiri/Desktop/ed.jpeg"
image = Image.open("ed.jpeg")

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
# let's only keep detections with score > 0.9
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.7)[0]

detected_objects = []
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    detected_objects.append(model.config.id2label[label.item()])
    print(detected_objects)

#for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
   # box = [round(i, 2) for i in box.tolist()]
    #print(
      #      f"Detected {model.config.id2label[label.item()]} with confidence "
        #    f"{round(score.item(), 3)} at location {box}"
    #)

import os
import openai

openai.api_key = ""

response = openai.Completion.create(
  model="text-davinci-003",
  prompt=f"Create a receipe from the edible items from this list: (detected_objects)",
  temperature=1,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

print(response["choices"][0]["text"])
