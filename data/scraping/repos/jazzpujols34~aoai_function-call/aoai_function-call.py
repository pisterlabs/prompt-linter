# Note: The openai-python library support for Azure OpenAI is in preview.
import os
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("AZURE_OPENAI_KEY")
openai.api_version = "2023-07-01-preview"
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")


messages= [
    {"role": "user", "content": "Find beachfront hotels in San Diego for less than $300 a month with free breakfast."}
]

functions= [  
    {
        "name": "search_hotels",
        "description": "Retrieves hotels from the search index based on the parameters provided",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The location of the hotel (i.e. Seattle, WA)"
                },
                "max_price": {
                    "type": "number",
                    "description": "The maximum price for the hotel"
                },
                "features": {
                    "type": "string",
                    "description": "A comma separated list of features (i.e. beachfront, free wifi, etc.)"
                }
            },
            "required": ["location"],
        },
    }
]  

response = openai.ChatCompletion.create(
    engine="gpt-4-32k",
    messages=messages,
    functions=functions,
    function_call="auto", 
)

print(response['choices'][0]['message'])
