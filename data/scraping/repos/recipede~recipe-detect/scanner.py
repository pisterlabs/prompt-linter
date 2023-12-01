from google.cloud import vision
import cohere  
import os
from dotenv import load_dotenv

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")

def is_food(flyer_text: str) -> bool:
    if COHERE_API_KEY == None:
        raise Exception("API key not found.")
    co = cohere.Client(COHERE_API_KEY)
    prompt = "The following is text from a grocery store flyer that sells conventional household goods and food. Determine if this item on the flyer is a food or not: " + flyer_text
    prompt += "\n\nPlease respond with only 'true' or 'false' based on whether the item is a food or not."
    response = co.generate(  
        model='command-nightly',  
        prompt = prompt,  
        max_tokens=200,  
        temperature=0.75)

    if response.generations == None:
        raise Exception("No response from API.")

    return response.generations[0].text.strip() == "true"

def extract_grocery(flyer_text: str) -> str:
    if COHERE_API_KEY == None:
        raise Exception("API key not found.")
    co = cohere.Client(COHERE_API_KEY)
    prompt = "The following is text from a grocery store flyer that sells conventional household goods and food. Determine what the product name is: " +flyer_text
    prompt += "\n\nPlease respond with only the name of the product." #kind of food or product that the item is."#"
    response = co.generate(  
        model='command-nightly',  
        prompt = prompt,  
        max_tokens=200,  
        temperature=0.75)

    if response.generations == None:
        raise Exception("No response from API.")

    return response.generations[0].text.strip()

def extract_flyer(image_uri: str) -> str:
    client = vision.ImageAnnotatorClient()
    response = None
    with open(image_uri, "rb") as image:
        file = image.read()
        byte_array = bytes(file)
        response = client.annotate_image({
        'image': {'content': byte_array },
        'features': [{'type_': vision.Feature.Type.TEXT_DETECTION}]
        })

    return str(response.text_annotations[0].description)

if __name__ == "__main__":
    flyer_text = str(extract_flyer("../grocery/crop_6.jpg"))
    print(flyer_text)
    print(extract_grocery(flyer_text))
    print(is_food(flyer_text))
