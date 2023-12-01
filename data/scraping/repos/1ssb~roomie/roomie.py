# Code by 1ssb
import os
import sys
import openai
from ultralytics import SAM, YOLO

def generate_prompt(words):
    prompt = "Only say the room name for which you have highest confidence and nothing else, like if you see a bed it is the bedroom, if you see a sink it's the kitchen. To regularise the expressions make sure you always output it as all small cap like: kitchen, bedroom, etc. In which room of the house is the embodied AI in, if these objects are found: "
    prompt += " , ".join(words)
    prompt += "?"
    return prompt

def get_places(prompt, openai_api_key):
    openai.api_key = openai_api_key
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
         {
      		"role": "user",
      		"content": "You are an embodied AI that takes in object inputs as text and tells the agent which room the agent is in and nothing else. Do not provide any other content, just the room name only."
         },
         {
                "role": "user",
                "content": prompt
         }
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    text = response['choices'][0]['message']['content']
    return text.strip()

def auto_annotate(data, det_model='yolov8x.pt', sam_model='sam_l.pt', device='cuda', output_dir=None):
    det_model = YOLO(det_model)
    sam_model = SAM(sam_model)
    det_results = det_model(data, stream=True, device=device, verbose=False)

    objects = []
    for result in det_results:
        class_ids = result.boxes.cls.int().tolist()
        if len(class_ids):
            boxes = result.boxes.xyxy  # Boxes object for bounding box outputs
            sam_results = sam_model(result.orig_img, bboxes=boxes, verbose=False, save=False, device=device)
            segments = sam_results[0].masks.xyn

            # Append the detected objects to the list
            objects.extend([det_model.names[c] for c in class_ids])
    return list(set(objects))

def process_images(folder, openai_api_key, det_model="yolov8x.pt", sam_model='sam_l.pt', device='cuda'):
    # Get the list of image files in the folder
    image_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Process each image file
    for image_file in image_files:
        # Annotate the image
        objects = auto_annotate(data=image_file, det_model=det_model, sam_model=sam_model, device=device)

        # Generate the prompt
        prompt = generate_prompt(objects)

        # Get the room name
        room = get_places(prompt, openai_api_key)

        # Print the room name
        print(f'The image:{image_file} is of the {room}.')

if __name__ == '__main__':
    # Set the folder and OpenAI API key
    folder = './images/'
    openai_api_key = 'sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'

    # Process the images in the specified folder
    process_images(folder, openai_api_key)

