import openai
import json
import requests
from config import OPENAI_API_KEY

# Replace 'your-api-key' with your actual OpenAI API key
GPT_MODEL = "gpt-3.5-turbo-0613"

def get_section_indices(message):
    # Define the headers for the API request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }

    # Define the function for extraction
    functions = [
        {
            "name": "get_sections",
            "description": "get the section numbers mentioned in the message. If no section numbers are listed then just return empty",
            "parameters": {
                "type": "object",
                "properties": {
                    "sections": {
                        "type": "array",
                        "description": "sections mentioned in array format",
                        "items": { "type": "number" },
                    },
                },
                "required": ["sections"]
            }
        }
    ]

    # Construct the messages for the conversation
    messages = [
        {"role": "system", "content": "You are a helpful assistant that scans a message and looks for indices"},
        {"role": "user", "content": message}
    ]

    # Make the API call
    try:
        response = requests.post(
            f"https://api.openai.com/v1/chat/completions",
            headers=headers,
            json={"model": GPT_MODEL, "messages": messages, "functions": functions}
        )
        response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code
    except requests.exceptions.RequestException as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return []

    # Parse the response to get the sections as integers
    try:
        assistant_message = json.loads(response.json()["choices"][0]["message"]["function_call"]["arguments"])
        section_strings = assistant_message["sections"]
        section_indices = [int(section) for section in section_strings]
        return section_indices
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        print("Error parsing the response or converting sections to integers")
        print(f"Exception: {e}")
        return []

