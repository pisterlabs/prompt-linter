from utils import prettify
from langchain.schema import BaseOutputParser
import requests
import logging


class UrlParser(BaseOutputParser):
    """Parse the URL to fetch a json output"""


    def parse(self, text: str):
        logging.info("Response received from LLM :"+ text)
        """Parse the output of an LLM call."""
        # Fetch the URL using the token 
        tokenFile = 'token.txt'
        file = open(tokenFile, 'r')
        token = file.read()
        file.close()
        if text == "Not found":
            return "Could not determine the URL %s" %text
        
        # Make the GET request with the bearer token
        headers = {
            'Authorization': f'Bearer {token}'
        }
        response = ""
        try:
            response = requests.get(text, headers=headers)
        except Exception as e:
            logging.error(f'Error: {str(e)}')
            logging.error ("\n\n Could not determine the URL %s" %text)
            raise ValueError("Could not determine the URL %s" %text)
        
        if response.status_code != 200:
            error = response.text
            logging.error("\n\nThe URL was: %(text)s. \n\nThe error response: %(error)s" %{"text":text, "error":prettify(error)})
            raise ValueError("\n\nThe URL was: %(text)s. The error response: %(error)s" %{"text":text, "error":error})
        
        logging.info("Response from the above URL: "+ prettify(response.text))
        return response.text 
