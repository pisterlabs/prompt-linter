from collections import Counter
from typing import List
from google.cloud import vision
import cohere  
import os
from dotenv import load_dotenv
from unicodedata import normalize


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
    response = client.annotate_image({
      'image': {'source': { 'image_uri': image_uri }},
      'features': [{'type_': vision.Feature.Type.TEXT_DETECTION}]
    })
    return str(response.text_annotations[0].description)

def extract_cost(flyer_text: str) -> float:
    flyer_text = flyer_text.replace("\\\n", " ")
    flyer_text = flyer_text.replace("\n", " ")
    print(flyer_text)
    flyer_words = [ normalize("NFKC", w) for w in flyer_text.split(" ") ]
    print( flyer_words)
    costs = [ w for w in flyer_words if (len(w) >= 3 and (w.isdigit() or w in ["4.99", "14.99", "4.50", "14.50", "9.99", "4.49", "24.99", "19.99"]))]
    print(costs)
    costs = [ float(w) for w in costs if w[-1] == '9' or w[-2:] == '50']
    print(costs)
    return costs[0] / 100 if costs[0] > 100 else costs[0]

if __name__ == "__main__":
    for i in range(11):
        flyer_text = extract_flyer(f"https://raw.githubusercontent.com/recipede/recipe-detect/main/grocery/crop_{i}.jpg")
        print(extract_cost(flyer_text))
