import openai
from serp import get_search_results
from webscrape import scrape_website
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the API keys
api_key = os.getenv("API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

#message here


def main():

    user_input = input("Enter desired fight: ")
    query = "Sherdog play by play " + user_input
    result = get_search_results(query)
    #scrape_website(result)

    file_path = 'outputs/result.txt'

    with open(file_path, 'r') as file:
        file_content = file.read()

    if result:
        # Convert the result to a string if it's not
        result_str = str(result)

        prompt = f"In detail, explain {user_input} fight using this following text file as information: {file_content}"
        response = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens = 1000 )
        print(response.choices[0].text.strip())

if __name__ == '__main__':
    main()
