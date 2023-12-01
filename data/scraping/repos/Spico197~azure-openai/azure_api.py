import os

import dotenv
from openai import AzureOpenAI


dotenv.load_dotenv()


def main():
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version="2023-10-01-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )

    messages = [
        {
            "role": "user",
            "content": "Find beachfront hotels in San Diego for less than $300 a month with free breakfast.",
        }
    ]

    functions = [
        {
            "name": "search_hotels",
            "description": "Retrieves hotels from the search index based on the parameters provided",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location of the hotel (i.e. Seattle, WA)",
                    },
                    "max_price": {
                        "type": "number",
                        "description": "The maximum price for the hotel",
                    },
                    "features": {
                        "type": "string",
                        "description": "A comma separated list of features (i.e. beachfront, free wifi, etc.)",
                    },
                },
                "required": ["location"],
            },
        }
    ]

    response = client.chat.completions.create(
        model="gpt-35-turbo-16k",
        messages=messages,
        functions=functions,
        function_call="auto",
    )

    print(response.choices[0].message.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
