# Importing the necessary libraries for the script
import requests # for making HTTP requests to the API
import os # for accessing the operating system
import openai # for interfacing with the OpenAI API
import argparse # for parsing command-line arguments

# Create an ArgumentParser object
parser = argparse.ArgumentParser()

# Add an argument to the parser for the prompt to send to the OpenAI API
parser.add_argument("prompt", help="The prompt to send to the OpenAI API")
parser.add_argument("file_name", help="Name of the file to save python script")

# Parse the arguments passed to the script
args = parser.parse_args()

# The API endpoint URL that we will use to make requests to the OpenAI API
api_endpoint = "https://api.openai.com/v1/completions"

# An empty API key for now, will be updated with the actual key
api_key = os.getenv("OPENAI_API_KEY")
# print(api_key) # debug

# Define the headers that will be sent in the API request
request_headers = {
    'Content-Type': "application/json",  # The content type of the request body
    'Authorization': "Bearer " + api_key  # The authorization token for the API request
}

# Define the data that will be sent in the API request
request_data = {
    'model': "text-davinci-003",  # The OpenAI model to use for generating the response
    'prompt': f"Write python script to {args.prompt}. provide only code, no text",  # The text prompt to generate a response to
    'max_tokens': 100,  # The maximum number of tokens (words or punctuation marks) to generate in the response
    'temperature': 0.5  # The level of randomness in the generated response, half creative - half predictable
}

# Sending a HTTP POST request to the API endpoint with the request headers and data
response = requests.post(api_endpoint, headers = request_headers, json = request_data)

# Check the HTTP response status code and write the JSON content if the status code is 200 in a python file
# Otherwise, print an error message with the actual status code returned by the server.
if response.status_code == 200:
    response_text = response.json()["choices"][0]["text"]
    with open(args.file_name, "w") as file:
        file.write(response_text)
else:
    print(f"\nRequest failed with status code: {str(response.status_code)}\n")
    
