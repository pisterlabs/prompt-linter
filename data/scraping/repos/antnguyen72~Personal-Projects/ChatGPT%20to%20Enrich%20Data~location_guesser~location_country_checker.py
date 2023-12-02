import json
import time
import openai

class LocationData:
    def __init__(self, countries_file_location: str, location_countries_file_location: str):
        self.countries_set = self.load_countries_data(countries_file_location)
        self.location_country_dict = self.load_location_countries_data(location_countries_file_location)

    @staticmethod
    def load_countries_data(file_location: str) -> set:
        """
        countries are stored in json with the format: {"countries":list["countries"]}

        return a set of countries to be used for lookup
        """
        countries_set = set([])

        with open(file_location,'r') as file:
            data = json.load(file)
            countries_set = set(data['countries'])

        return countries_set

    @staticmethod
    def load_location_countries_data(file_location: str) -> dict:
        """
        location_country data is stored in json with the format: {"location":"countries"}

        return a dict of location_countries to be used for lookup
        """
        location_country_dict = dict()

        with open(file_location,'r') as file:
            location_country_dict = json.load(file)
        
        return location_country_dict

class LocationChecker:
    def __init__(self, openai_apikey: str, countries_set: set, location_country_dict: dict, country_list: list):
        self.openai_apikey = openai_apikey
        self.countries_set = countries_set
        self.location_country_dict = location_country_dict
        self.country_list = country_list

        # Initialize the OpenAI API key
        openai.api_key = openai_apikey

    def call_openai(self, prompt: str) -> str:
        """
        Call the OpenAI API and return the generated text.

        Args:
            prompt (str): The prompt to generate text from.

        Returns:
            str: The generated text.
        """
        try:
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                temperature=0,
                max_tokens=60,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            return response.choices[0]['text'].strip()
        except openai.error.RateLimitError:
            # Catch the rate limit error and wait for one minute before trying again
            print("Rate limit reached. Waiting for 60 seconds before trying again...")
            time.sleep(60)
        except Exception as e:
            # Catch any exceptions thrown by the OpenAI API and print the error message
            print(f"Error calling OpenAI API: \nError Type: {type(e)}\nError Message: {e}")
            return "Error calling OpenAI API"

    def check_location(self, location: str) -> str:
        """
        Check which country a given location is in.

        Args:
            location (str): The location to check.

        Returns:
            str: The name of the country the location is in, or "Not in given list" if it is not in the list.
        """
        possible_country_data = location.split(', ')[-1]

        if possible_country_data is not None and possible_country_data in self.countries_set:
            return possible_country_data
        elif location in self.countries_set:
            return location
        elif location in self.location_country_dict:
            return self.location_country_dict[location]
        else:
            print(f"\nNew location detected --> {location}")
            prompt = f"Which country is {location} in? Give answer based on the following list: {self.country_list} if not in the list, return 'Not in given list'"
            country = self.call_openai(prompt)
            print(f"Chatgpt found location to be in country --> {country}, updating database")
            self.location_country_dict[location] = country
            if str(country) == "Not in given list":
                return None
            else:
                return country