import os
import dotenv
import openai

dotenv.load_dotenv()
openai.api_key  = os.getenv('OPENAI_API_KEY')


def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(model=model, messages=messages, temperature=0)
    # temperature: the degree of randomness of the model's output
    ret = response.choices[0].message["content"]
    return ret


fact_sheet_chair = """
OVERVIEW
- Part of a beautiful family of mid-century inspired office furniture, including filing cabinets, desks, bookcases, meeting tables, and more.
- Several options of shell color and base finishes.
- Available with plastic back and front upholstery (SWC-100) or full upholstery (SWC-110) in 10 fabric and 6 leather options.
- Base finish options are: stainless steel, matte black, gloss white, or chrome.
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


prompt = f"""
Your task is to help a marketing team create a description for a retail website of a product based on a technical fact sheet.

Write a product description based on the information provided in the technical specifications delimited by triple backticks.

Technical specifications: ```{fact_sheet_chair}```
"""
response = get_completion(prompt)
print(response)
'''
Introducing our stunning mid-century inspired office chair, the perfect addition to any home or business setting. Part of a beautiful family of office furniture, this chair is available in several shell color and base finish options, allowing you to customize it to your liking. Choose between plastic back and front upholstery or full upholstery in a variety of fabric and leather options.

Constructed with a 5-wheel plastic coated aluminum base and a pneumatic chair adjust for easy raise/lower action, this chair is both durable and functional. It also comes with the option of soft or hard-floor casters and two choices of seat foam densities: medium or high.

The chair is available with or without armrests, and the armrests come in 8 different positions. The base finish options are stainless steel, matte black, gloss white, or chrome. The chair is also qualified for contract use.

With a width of 53 cm, depth of 51 cm, and height of 80 cm, this chair is the perfect size for any office space. The seat height is 44 cm and the seat depth is 41 cm.

The materials used in the construction of this chair are of the highest quality. The shell base glider is made of cast aluminum with modified nylon PA6/PA66 coating and has a shell thickness of 10 mm. The seat is made of HD36 foam.

This chair is made in Italy and is sure to add a touch of elegance to any workspace.
'''

# it's too long, let's try again
prompt = f"""
Your task is to help a marketing team create a description for a retail website of a product based on a technical fact sheet.

Write a product description based on the information provided in the technical specifications delimited by triple backticks.

Use at most 50 words.

Technical specifications: ```{fact_sheet_chair}```
"""
response = get_completion(prompt)
print(response)
'''Introducing our mid-century inspired office chair, perfect for home or business settings. With a range of shell colors and base finishes, and the option of plastic or full upholstery in various fabrics and leathers,
 this chair is both stylish and comfortable. The 5-wheel base and pneumatic chair adjust make it easy to use, while the choice of soft or hard-floor casters and armrest options allow for customization. Made in Italy with high-quality materials, this chair is qualified for contract use.'''
