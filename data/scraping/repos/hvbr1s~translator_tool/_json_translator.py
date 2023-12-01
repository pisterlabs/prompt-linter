import asyncio
import os
import openai
import json
from doctran import Doctran
from dotenv import load_dotenv

load_dotenv()

env_vars = [
    'OPENAI_API_KEY',
]
os.environ.update({key: os.getenv(key) for key in env_vars})
openai.api_key = os.getenv('OPENAI_API_KEY')

doctran = Doctran(openai_api_key=openai.api_key)

# define an async function
async def translate_doc(input_json_file, output_json_file):
    with open(input_json_file, "r") as read_file:
        data = json.load(read_file)

    translated_data = []

    for item in data:
        document = doctran.parse(content=item["text"])
        translated = await document.translate(language="french").execute()
        translated_text = translated.transformed_content
        print(translated_text)
        
        new_item = dict(item)  # create a copy of the item
        new_item["text"] = translated_text  # replace "text" field with translated text
        translated_data.append(new_item)
    
    with open(output_json_file, "w") as write_file:
        json.dump(translated_data, write_file, ensure_ascii=False, indent=4)

# use the asyncio.run() function to run the async function
asyncio.run(translate_doc("input.json", "translated.json"))
