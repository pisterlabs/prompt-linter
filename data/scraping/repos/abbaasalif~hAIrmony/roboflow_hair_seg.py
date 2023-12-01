from roboflow import Roboflow
import base64
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import os
from openai import OpenAI
import urllib.request
import cv2
style = {
    "fringe": "A fringe haircut is characterized by a section of hair cut to fall over the forehead, framing the face. It varies in length, texture, and style. Length-wise, it can range from just above the eyebrows to longer styles that touch the eyes. Textures vary, with options for a blunt, straight-across cut or textured layers that soften the look. The width can span the entire forehead or be narrower, leaving more forehead visible. Shapes can be straight, curved, or asymmetrical to complement different facial features.",
    "taper": "The taper hairstyle for men is characterized by gradual length variation, starting shorter at the sides and back and gradually increasing in length toward the top. This gradual transition creates a smooth and blended look. The sides and back are typically cut shorter, often with clippers, starting with a shorter guard size and gradually increasing in length as it moves upward. This creates a fade effect, with the hair seamlessly tapering from shorter to longer. The transition can be a low, mid, or high taper, depending on where the fading begins. The top of the hair can vary in length, allowing for versatility in styling. Some prefer to keep it longer for more styling options, while others opt for a shorter length for easy maintenance. The top can be styled in various ways, from textured crops to slicked-back looks, providing versatility for different preferences and face shapes.",
    "quiff": "The quiff haircut is a stylish men's hairstyle defined by its voluminous, swept-back top. It features shorter sides and back, gradually transitioning to longer hair on top. The hallmark of the quiff is the prominent volume and height at the front, achieved through lifting and styling the hair upward and back from the forehead. Typically, the longer hair at the top front is styled upward and backward, creating a voluminous, textured appearance. This lift and height are achieved by blow-drying the hair while using styling products like pomade, wax, or gel to maintain the shape and hold. The sides and back can vary in length, often tapered or faded to add contrast and emphasis to the voluminous top. The top section's length provides versatility, allowing for various styling options—some prefer a longer, more dramatic quiff, while others opt for a more subtle, shorter variation.",
    "mohawk": "A Mohawk hairstyle is defined by a central strip of hair running from the forehead to the nape of the neck, with shaved or closely cropped sides. The central strip, known as the crest or crest-like portion, is typically longer and stands upright, extending from the forehead to the back of the head. This strip can vary in width and length, allowing for personalization in style. The hair on the sides of the head is shaved or trimmed very short, creating a stark contrast with the prominent central strip. The sides can be completely shaved, faded, or have a graduated length from the central strip to the shaved portions. Styling the central strip often involves using hair products such as gel, wax, or pomade to shape and hold the hair upright. The Mohawk style offers versatility in the length and texture of the crest, allowing for variations such as spiked, textured, or braided styles.",
}
img_path = r'./hair_samples/PXL_20231118_200015682.jpg'
img_path = 'temp.jpg'
prompt = 'There is a man getting a haircut. The side is very short skin fade. The texture of the hair is straight'
rf = Roboflow(api_key="eJ3QeQPzbAdH0XlMNzcs")
project = rf.workspace().project("color-hair-eyebrow-lips")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
model = project.version(1).model

#get image from webcam
cap = cv2.VideoCapture(0)
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite('temp.jpg', frame)
        cv2.destroyAllWindows()
        break
output = model.predict(img_path).json()
encoded_segmentation_mask = output['predictions'][0]['segmentation_mask']

# Decode the base64 string
decoded_img = base64.b64decode(encoded_segmentation_mask)
img = Image.open(io.BytesIO(decoded_img))

# Convert to numpy array
mask_array = np.array(img)

# Assume the class you are interested in is represented by a specific pixel value
# For example, if 'Eyebrow' is represented by the pixel value 1
class_value = 2  # Update this with the actual value
binary_mask = (mask_array == class_value).astype(int)

# Saving the binary mask as an image
mask_image = Image.fromarray((binary_mask * 255).astype(np.uint8))

#resize the mask image to the original image size
original_image = Image.open(img_path)
#convert to RGBA
original_image = original_image.convert("RGBA")
original_array = np.array(original_image)
mask_image = mask_image.resize(original_image.size)
mask_array = np.array(mask_image)
original_array[mask_array == 255] = (0, 0, 0, 0)
result_image = Image.fromarray(original_array)
width, height = result_image.size
new_size = (1024, 1024)
result_image = result_image.resize(new_size, Image.LANCZOS)
result_image.save('temp_result.png')
original_image = original_image.resize(new_size, Image.LANCZOS)
original_image.save('temp_original.png')
# cv2.imshow('temp_result.png', np.array(result_image))
user_style = input("Enter the style you want.\n\tfringe\n\ttaper\n\tquiff\n\tmohawk\n")

if user_style in style:
    prompt = 'There is a person with a particular hairstyle. The hairstyle is a' + user_style + ' ' + style[user_style]
response = client.images.edit(
    model='dall-e-2',
    prompt=prompt,
    image=open('temp_original.png',"rb"),
    mask=open('temp_result.png',"rb"),
    n=4,
    size='1024x1024',

)
print(response.data[0].url)
#download the image
imgs = []
for i in range(4):
    urllib.request.urlretrieve(response.data[i].url, f"generated_{i}.png".format(i))
    imgs.append(cv2.imread(f"generated_{i}.png".format(i)))

#show the images
for i in range(4):
    cv2.imshow('generated_'+str(i)+'.png', imgs[i])
cv2.waitKey(0)
cv2.destroyAllWindows()
