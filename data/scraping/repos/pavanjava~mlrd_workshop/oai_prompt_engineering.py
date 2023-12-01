import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file


class OpenAIPromptEngineeringUtility:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def get_response(self, prompt, model='gpt-4-1106-preview'):
        messages = [{"role": "user", "content": prompt}]
        resp = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,  # this defines the randomness of the model response,
            max_tokens=256
        )

        return resp.choices[0].message.content


if __name__ == "__main__":
    obj = OpenAIPromptEngineeringUtility()

    fact_sheet_chair = """- Part of a beautiful family of mid-century inspired office furniture,
        including filing cabinets, desks, bookcases, meeting tables, and more.
        - Several options of shell color and base finishes.
        - Available with plastic back and front upholstery (SWC-100)
        or full upholstery (SWC-110) in 10 fabric and 6 leather options.
        - Base finish options are: stainless steel, matte black,
        gloss white, or chrome.
        - Chair is available with or without armrests.
        - Suitable for home or business settings.
        - Qualified for contract use.
        
        CONSTRUCTION
        - 5-wheel plastic coated aluminum base.
        - Pneumatic chair adjust for easy raise/lower action.
        
        DIMENSIONS
        - WIDTH 53 CM | 20.87”
        - DEPTH 51 CM | 20.08”
        - HEIGHT 80 CM | 31.50”
        - SEAT HEIGHT 44 CM | 17.32”
        - SEAT DEPTH 41 CM | 16.14”
        
        OPTIONS
        - Soft or hard-floor caster options.
        - Two choices of seat foam densities:
         medium (1.8 lb/ft3) or high (2.8 lb/ft3)
        - Armless or 8 position PU armrests
        
        MATERIALS
        SHELL BASE GLIDER
        - Cast Aluminum with modified nylon PA6/PA66 coating.
        - Shell thickness: 10 mm.
        SEAT
        - HD36 foam
        
        COUNTRY OF ORIGIN
        - Italy
    """
    product_review = """ """

    prompt = f"""
    Your task is to help a marketing team create a 
    description for a retail website of a product based 
    on a technical fact sheet.
    
    Write a product description based on the information 
    provided in the technical specifications delimited by 
    triple backticks.
    
    Technical specifications: ```{fact_sheet_chair}```
    """
    response = obj.get_response(prompt)
    print(response)

    prompt = f"""
    Your task is to generate a short summary of a product \
    review from an ecommerce site. 
    
    Summarize the review below, delimited by triple 
    backticks, in at most 30 words. 
    
    Review: ```{product_review}```
    """

    response = obj.get_response(prompt)
    print(response)
