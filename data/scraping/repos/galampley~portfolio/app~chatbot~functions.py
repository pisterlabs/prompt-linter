import os
import openai
import requests
import matplotlib.pyplot as plt
import time
from bs4 import BeautifulSoup

from dotenv import load_dotenv

dotenv_path = '/Users/galampley/Documents/secrets.env'  # Replace with the path to your .env file if it's not in the current directory
load_dotenv(dotenv_path)

env_variable = os.getenv('OPENAI_API_KEY')

def list_files(assistant_id):
    # Construct the URL to list files associated with the assistant
    files_url = f"https://api.openai.com/v1/assistants/{assistant_id}/files"
    
    # Set up headers with your API key
    headers = {
        "Authorization": f"Bearer {env_variable}",
        "OpenAI-Beta": "assistants=v1",
    }
    
    try:
        # Make a GET request to retrieve the list of files
        response = requests.get(files_url, headers=headers)
        
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            files = response.json().get("data", [])
            # Print the list of files
            print("List of files associated with the assistant:")
            for file in files:
                print(f"File ID: {file['id']}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error: {str(e)}")

def fetch_and_display_run_steps(thread_id, run_id):
    # Construct the API endpoint URL
    url = f"https://api.openai.com/v1/threads/{thread_id}/runs/{run_id}/steps"

    # Set up headers with your API key
    headers = {
            "Authorization": f"Bearer {env_variable}",
            "OpenAI-Beta": "assistants=v1",
        }

    try:
        # Make the GET request to fetch run steps
        response = requests.get(url, headers=headers)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the JSON response
            run_steps = response.json()

            print("Run Steps:")
            for step in run_steps["data"]:
                print(step)
        else:
            print(f"Error: {response.status_code} - {response.text}")

    except Exception as error:
        print(error)

def download_resume():
    file_id = '1Jwds5kBw8GGHFPOAM9mFooGtB3LrAstV'
    download_url = f'https://drive.google.com/uc?export=download&id={file_id}'

    # The local file path where you want to download the PDF
    local_file_path = 'app/reference_content/resume.pdf'

    # Perform the request and download the file
    response = requests.get(download_url)
    if response.status_code == 200:
        with open(local_file_path, 'wb') as f:
            f.write(response.content)
    else:
        print('Failed to download the file.')

def download_about_me():
    # Path to your local HTML file
    html_file_path = 'app/templates/index.html'

    # Path to the text file you will write to
    text_file_path = 'app/reference_content/about_me.txt'

    # Read the HTML content from the file
    with open(html_file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()

    # Parse the HTML with Beautiful Soup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find the 'about-me' section and extract the text
    about_me_section = soup.find('section', id='about-me')
    about_me_text = about_me_section.get_text(separator='\n', strip=True)

    # Write the extracted text to a text file
    with open(text_file_path, 'w', encoding='utf-8') as file:
        file.write(about_me_text)


def get_medium_user_id(username, rapidapi_key):
        url = f"https://medium2.p.rapidapi.com/user/id_for/{username}"
        headers = {
            "x-rapidapi-key": rapidapi_key
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            user_id = response.json().get("id")
            return user_id
        else:
            raise Exception(f"Error fetching user ID: {response.status_code}")

def get_article_ids(user_id, rapidapi_key):
        url = f"https://medium2.p.rapidapi.com/user/{user_id}/articles"
        headers = {
            "x-rapidapi-key": rapidapi_key
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()["associated_articles"]
        else:
            raise Exception(f"Error fetching articles: {response.status_code}")

def get_article_info(article_id, rapidapi_key):
                url = f"https://medium2.p.rapidapi.com/article/{article_id}"
                headers = {"x-rapidapi-key": rapidapi_key}
                response = requests.get(url, headers=headers)
                
                if response.status_code == 200:
                    article_data = response.json()
                    return article_data
                else:
                    raise Exception(f"Error fetching article content: {response.status_code}")

def get_article_content(article_id, rapidapi_key):
                url = f"https://medium2.p.rapidapi.com/article/{article_id}/content"
                headers = {
                    "x-rapidapi-key": rapidapi_key
                }
                response = requests.get(url, headers=headers)
                if response.status_code == 200:
                    return response.json()["content"]
                else:
                    raise Exception(f"Error fetching article content: {response.status_code}")

def download_blog(username, rapidapi_key, directory='app/reference_content/blog'):
    # First, obtain the user ID
    user_id = get_medium_user_id(username, rapidapi_key)
    
    # Then, get all article IDs for that user
    article_ids = get_article_ids(user_id, rapidapi_key)
    
    # Get the number of articles already downloaded
    files_in_directory = os.listdir(directory)
    number_of_blogs = len([file for file in files_in_directory if os.path.isfile(os.path.join(directory, file))])
    
    # If the number of articles matches the number of files, stop the function
    if len(article_ids) == number_of_blogs:
        print("All articles have already been downloaded.")
        return

    # Download articles that haven't been downloaded yet
    for article_id in article_ids:
        try:
            info = get_article_info(article_id, rapidapi_key)
            file_name = f"{info['title'].replace('/', '_')}.txt"  # Replace / to avoid path issues
            file_path = os.path.join(directory, file_name)
            
            if not os.path.isfile(file_path):
                content = get_article_content(article_id, rapidapi_key)
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(content)
                print(f"File created: {file_name}")
            else:
                print(f"File already exists: {file_name}, skipping...")
        except Exception as e:
            print(f"Failed for {article_id}: {e}")


