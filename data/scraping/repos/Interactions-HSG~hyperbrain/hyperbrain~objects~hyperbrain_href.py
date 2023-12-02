"""
2023-bt-raffael-rot
19-607-928
Prof. Dr. Simon Mayer
Danai Vachtsevanou, MSc.
"""

# Import utilities
import json
import requests
from urllib.parse import unquote
import datetime
from bs4 import BeautifulSoup
import re


class HyperBrainHref:
    """
    This is HyperBrain. It enables Hypermedia Guidance.
    """

    def __init__(self) -> None:
        """ Constructor
        """
        # Get the API KEY from a secure .txt file
        with open('hyperbrain/data/API_KEY.txt', 'r') as f:
            self.API_KEY = f.read()
        self.API_URL = "https://api.openai.com/v1/chat/completions"  # API URL from OpenAI
 

    @staticmethod
    def _set_logs(log: str) -> int:
        """
        :param log: The log entry to save in the log file.
        :return: 0
        """

        now = datetime.datetime.now()  # Get the real time
        current_time = now.strftime("%H:%M:%S")  # Formatting of the date

        date_log = f"[{current_time}]  {log}\n"  # Append log entry to instance variable

        # Append a new line to the log file
        with open('hyperbrain/data/hyperbrain_logs.txt', 'a') as file:
            file.write(f"{date_log}")

        return 0

    @staticmethod
    def _get_hypermedia_data(url: str) -> list:
        """ GET Request
        This method requests the html/text data of the Hypermedia. It extracts the hypermedia references.
        :param url: URL
        :return: List of links
        """

        response = requests.get(url)  # GET request

        html_doc = BeautifulSoup(response.content, 'html.parser')  # Transforming context into bs4

        hrefs = []  # Init an empty array for the hypermedia links
        titles = []

        for paragraph in html_doc.find_all('p'):
            # Get all hypermedia references
            for link in paragraph.find_all('a'):

                if link.get("title"):
                    hrefs.append(link.get('href'))
                    titles.append(link.get("title"))

        log_entry = f"The context of hypermedia environtment '{url}' " \
                    f"was successfully downloaded and extracted."  # Init a new log entry

        HyperBrainHref._set_logs(log_entry)  # Write a new log entry

        return hrefs, titles

    def _ask(self, query, model="gpt-3.5-turbo", temperature=0.9):
        """
        :param query: Query is the input for the LLM.
        :param model: Select the LLM model from OpenAI.
        :param temperature: Hyperparameter of the LLM to set the randomness.
        :return: Return the response of the LLM.
        """
        # Init the headers for the request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.API_KEY}"
        }

        # Init the data for the request
        data = {
            "model": model,
            "messages":
                [
                    {
                        "role": "system",
                        "content": "You are a helpful system to find the most likely related keyword based on the given list."
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
            "temperature": temperature
        }

        self._set_logs("Query: " + query)

        response = requests.post(self.API_URL, headers=headers, data=json.dumps(data))  # POST request to the OpenAi API

        print(response)
        print(response.json())

        exit()

        data = response.json()  # Get the JSON data from the respone

        result = data['choices'][0]['message']['content']  # Init the result of the request

        log_entry = f"The request for the question was executed successfully."  # Init a log entry

        self._set_logs(log_entry)  # Write a log entry

        return result  # Return result

    def hyperbrain(self, keyword: str, entry_point: str) -> str:
        """
        :param keyword: High-level goal
        :param entry_point: Starting resource of an API «Root Resource»
        :return: Return the URI of the high-level goal.
        """

        self._set_logs(log=f"{10 * '*'}\nStart HyperBrain for {keyword}")  # Start

        found = True  # Condition for the while loop

        while found:  # Search until HyperBrain finds the answer

            # Query to check if the entry point is the right hypermedia link

            query = f"Available link: '{entry_point}'\n" \
                    f"Guess if the available link is the hypermedia reference for {keyword}. Return with TRUE/FALSE."

            answer = self._ask(query)  # Get an answer‚

            self._set_logs(log=f"Answer: {answer}")  # Write a log entry

            # If True, the final application state is found
            if "TRUE" in answer:
                found = False

            else:  # If he does not find the answer, he shows the most related link to the topic

                hrefs, titles = self._get_hypermedia_data(entry_point)  # Get hypermedia context

                y = 200  # Number of links

                self._set_logs(log=f"Available Links: {hrefs[:y]}")  # Write a log entry

                # Query 2
                query = f"List of keywords: '{titles[:y]}'. " \
                        f"Guess which of these keywords is most likely related to '{keyword}'. " \
                        f"Provide a keyword and one sentence explanation."

                answer = self._ask(query)  # Ask which link is most related

                self._set_logs(log=f"Answer: {answer}")  # Write a log entry

                # Find all links
                matches = re.findall(r'(["\'])(.*?)\1', answer)
                matches = [match[1] for match in matches if match[1]]

                if len(matches) > 1:  # If several keywords were found, delete the high-level goal from the list
                    matches.remove(keyword)
                else:  # If there is only the high-level goal, start the next loop
                    print("Not found")
                    continue

                answer = matches[0]  # Get the first keyword from the list

                answer = "https://en.wikipedia.org/wiki/" + answer

                print(f"Answer: {answer}")

                entry_point = unquote(answer)  # Set the new url as the new local hypermedia environment

                continue

        self._set_logs(f"End HyperBrain \n{10 * '*'}")

        return answer
