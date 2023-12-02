from dotenv import load_dotenv
load_dotenv()

import time 


import requests
from bs4 import BeautifulSoup

import os
import openai

api_key = os.getenv("API_KEY7")


    
def data_scrape(url, timeout=2):
    try:
        start_time = time.time()
        response = requests.get(url, timeout=timeout)
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        if response.status_code == 200:
            print("Successful response")
            if elapsed_time > timeout:
                print("Request took more than 2 seconds. Skipping...")
                return -1
            soup = BeautifulSoup(response.text, 'html.parser')
            return soup.text
        else:
            print('Failed to retrieve the page. Status code:', response.status_code)
            return -1
    except requests.exceptions.Timeout:
        print('Request timed out after', timeout, 'seconds. Skipping...')
        return -1
    except Exception as e:
        print('An error occurred:', str(e))
        return -1




def search_websites_with_keyword(keyword):
    # Define the search query
    search_query = f"intitle:{keyword}"  # This query searches for "rpi" in the title of web pages

    # Send a GET request to Google Search with the query
    search_url = f"https://www.google.com/search?q={search_query}"
    headers = {"User-Agent": "Your User Agent Here"}  # Replace with your User-Agent header
    response = requests.get(search_url, headers=headers)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content of the search results page
        soup = BeautifulSoup(response.text, "html.parser")

        # Find and extract search result links
        search_results = soup.find_all("a")
        for result in search_results:
            link = result.get("href")
            if link and link.startswith("/url?q="):
                # Extract the actual URL from the Google search result link
                url = link[7:]  # Remove "/url?q=" prefix
                info = data_scrape(url)
                if (info != -1):
                    message_list.append( {"role": "system", "content": info})

                



                # You can further process this URL or send a request to scrape data from the website

    else:
        print("Failed to retrieve search results.")







def create_response(user_input, message_list=[]):
    time.sleep(1)
    message_list.append( {"role": "user", "content": user_input})
    print(message_list)
    openai.api_key = api_key
    print("starting creation")
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=message_list,
    temperature = 2 
    )
    print("created")
    print(completion.choices[0].message.content)
    message_list.pop()
    
    return completion.choices[0].message.content



if __name__ == "__main__":
    message_list=[
    {"role": "system", "content": "You are a helpful assistant."},
  ]
    user_input = input("Enter a message: ")
    search_websites_with_keyword("rpi")

    ex_url = "https://www.rpi.edu/"
    some_info = data_scrape(ex_url)
    message_list.append( {"role": "system", "content": some_info})

    create_response(user_input, message_list)
