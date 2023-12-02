import cohere
from PIL import Image
import io
import requests

API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
headers = {"Authorization": "Bearer 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"}

co = cohere.Client("Poc7e6ygMiSArhWVvxr6igd1wWHhKUWY2b1n3xuL")

def generate(prompt):
    response = co.generate(  
    model='command-nightly',  
    prompt = prompt,  
    max_tokens=200,  
    temperature=0.750)

    result= response.generations[0].text 
    return result 

def prompt(body_type, description):
    text = "The client is female with {} body type, and she is seeking recommendation from a personal stylist for pants and shoes that would go well with a {}. As advice coming from a stylist, it should align with fashion trends, color theory, and other concepts to make a fashionable combination. She expects recommendation like dark skinny jeans goes well with cream-colored oversized sweater. Monochrome or complementary colors are also good suggestions. The color of the items should be specified and matched tastefully. The two recommendations should be described by 5 words or less and separated by a newline in the format: 'Type of pants: pants description \n Type of shoes: shoe description".format(body_type, description)
    return text

def generate_image(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.content

def store_image(type, description): 
  try: 
    image_bytes = generate_image({
      "inputs": "Generate a professional-grade image of a" + description + "against a neutral background. Create a visually appealing composition that highlights the [clothing or shoe]'s features, making it highly desirable for women. Any peron's face should be cropped out of the image."
    })
  except: 
    print("Retry")

  image = Image.open(io.BytesIO(image_bytes))
  
  image.save(type + ".jpg")

# result = generate(prompt("pear-shaped", "Leather Biker Jacket"))
# pants_desc = result.split('\n')[0]
# shoes_desc = result.split('\n')[1]
# print("1: {}".format(pants_desc))
# print("2: {}".format(shoes_desc))

# store_image("pants", pants_desc)
# store_image("shoes", shoes_desc)