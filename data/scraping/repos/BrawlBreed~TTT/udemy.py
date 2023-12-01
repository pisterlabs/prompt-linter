import os
import openai
from chrome import chrome_setup
import undetected_chromedriver as webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains, Keys
from selenium.webdriver.support.select import Select
from selenium.common.exceptions import NoSuchElementException, WebDriverException
# Add the chatGPT actions module to translate the JSON generation on a already trained model in ChatGPT 3.5
import time
from dotenv import load_dotenv

class Udemy():
    def __init__(self):
        self.driver = chrome_setup()
        
    def logIn(self):
        self.driver.get('https://www.udemy.com/')
        time.sleep(999)

class OpenAi():
        def __init__(self):
            load_dotenv()
            self.openAiKey = os.getenv('OPENAI_API_KEY')
            openai.organization = "org-nIi2D0aBjyBSXxRJbmdtPN24"
            openai.api_key = self.openAiKey

        def getPrompt(self, message):
            response = openai.Completion.create(
            engine="gpt-3.5-turbo",
            prompt="Translate the following English text to Bulgarian: 'Hello World!'",
            max_tokens=60
            )

            print(response)

        

# udemy = Udemy()
# openAi = OpenAi()
# openAi.getPrompt('Hello World!')

def getPrompt(message):
    load_dotenv()
    openAiKey = os.getenv('OPENAI_API_KEY')
    openai.organization = "org-nIi2D0aBjyBSXxRJbmdtPN24"
    openai.api_key = openAiKey
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Translate the text in Bulgarian, while doing so keep the JS context words with their English form."},
            {"role": "user", "content": message},
        ],   
        # Defining the max tokens in the Udemy transcript
        max_tokens=60
    )

    return response["choices"][0]["message"]["content"]

message = getPrompt('Quote me Julius Ceaser')
print(message)