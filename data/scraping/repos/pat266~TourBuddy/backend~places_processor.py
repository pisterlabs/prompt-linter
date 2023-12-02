from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool, AgentExecutor
from duckduckgo_search import DDGS
from scrape_and_summarize import ScrapeAndSummarize
from config import Config
from opentripmap import OpenTripMapAPI
import json

class PlacesProcessor:
    def __init__(self, client):
        self.client = client
        self.ddgs = DDGS()
        self.scrapeAndSummarize = ScrapeAndSummarize()
        self.mrkl = self.initialize_langchain_agent()
        self.opentripmap = OpenTripMapAPI()
        

    def get_summarized_reviews(self, place):
        all_reviews = self.scrapeAndSummarize.ddgsearch(place, 3)
        return self.scrapeAndSummarize.summarize_reviews(all_reviews, None)

    def initialize_langchain_agent(self):
        turbo_llm = ChatOpenAI(api_key=Config.OPENAI_API_KEY, temperature=0, model="gpt-3.5-turbo-1106") # 16k tokens sent, 4k tokens received
        tools = [
            Tool(
                name="get_place_information",
                func=self.get_summarized_reviews,
                description="Get summarized reviews of a place"
            ),
        ]

        return initialize_agent(
            agent=AgentType.OPENAI_FUNCTIONS,
            tools=tools,
            llm=turbo_llm,
            verbose=False,
            max_iterations=2,
            early_stopping_method='generate',
        )

    def get_recommended_places(self, latitude="33.771030", longitude="-84.391090", radius=10):
        keys_to_keep = ['name', 'address', 'latitude', 'longitude', 'phone', 'preference', 'id']
        preferences_list = ['sports', 'art and culture', 'museum and history', 
                            'food and dining', 'nature and outdoors', 'music', 
                            'technology', 'shopping', 'movies and entertainment']
        recommended_places_list = []

        for preference in preferences_list:
            for original_dict in self.ddgs.maps(f"places related to {preference}", 
                                           latitude=str(latitude), 
                                           longitude=str(longitude), 
                                           radius=radius, 
                                           max_results=10):
                # Add the 'preference' key and value directly to the original dictionary
                original_dict['preference'] = preference
                # add an id to distinguish between different places
                original_dict['id'] = str(original_dict['latitude']) + str(original_dict['longitude'])
                # replace the 'title' key with 'name' key
                original_dict['name'] = original_dict.pop('title', None)
                recommended_places_list.append({k: original_dict[k] for k in keys_to_keep if k in original_dict})
        return recommended_places_list
    
    def get_recommended_places_open_trip_map(self, latitude="33.771030", longitude="-84.391090", radius=10, preferences=['banks', 'restaurants']):
        recommended_places_list = self.opentripmap.nearby_search(latitude, longitude, radius * 1000, kinds=preferences)
        # print(recommended_places_list)
        # Filter out dictionaries where 'name' is empty or null
        recommended_places_list = [place for place in recommended_places_list if place.get('name')][:20]
        # print('Got the recommended places from OpenTripMap API')
        for recommended_place in recommended_places_list:
            # print(recommended_place)
            recommended_place['id'] = str(recommended_place['point']['lat']) + str(recommended_place['point']['lon'])
            recommended_place['latitude'] = recommended_place['point']['lat']
            recommended_place['longitude'] = recommended_place['point']['lon']
            recommended_place['kinds'] = self.clean_kinds(recommended_place['kinds'])
            recommended_place.pop('point', None)
            recommended_place.pop('rate', None)
            recommended_place.pop('osm', None)
            recommended_place.pop('dist', None)

        return recommended_places_list

    def generate_informations(self, place_name):
        prompt = f"""
        Based on what you know, generate information about this place: {place_name} and also get more information by using the tool to get summarized reviews of that place.
        Key information such as the environment and atmosphere of the place. 
        If possible, estimate the range of the cost, and give some recommendations of what food people order if it is a restaurant or common activities people do in here.
        Label them appropriately, and go to the next line for each detail.
        """
        return self.mrkl.run(prompt)
    
    def generate_information(self, place_name):
        prompt = f"""
        Based on what you know, generate about this place: {place_name}. If you do not know anything, simply return "".
        Key information such as the environment and atmosphere of the place.
        If possible, estimate the cost (preferably range of number), and give some recommendations of what food people order if it is a restaurant, or common activities people do in here.
        Label them appropriately, and go to the next line for each detail.
        """
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            max_tokens=500,
            temperature=0,
            messages=
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        info = response.choices[0].message.content.strip()
        return info
    
    def process_places(self, places):
        # print(places)
        promptsArray = []
        for place in places:
            place['info'] = "" # generate a placeholder for the information
            promptsArray.append(f"""
                Based on what you know, generate about this place: {place['name']} with the ID: {place['id']}.
                Key information such as the environment and atmosphere of the place.
                If possible, estimate the cost (preferably range of number), and give some recommendations of what food people order if it is a restaurant, or common activities people do in here.
                Label them appropriately, and go to the next line for each detail.
            """)
        stringifiedPromptsArray = json.dumps(promptsArray)
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            # max_tokens=500,
            temperature=0,
            response_format= {"type": "json_object"},
            messages=
            [
                {"role": "system", "content": "You are a helpful assistant. Reply in JSON format to all answers with as much details as possible to each query. Use the ID as the key to the JSON."},
                {"role": "user", "content": stringifiedPromptsArray},
            ],
        )
        # print(response.choices[0].message.content)
        batchCompletion = json.loads(response.choices[0].message.content)
        
        # process the batch responses
        for place in places:
            data = batchCompletion[place['id']]
            stringified_data = self.stringify_dictionary(data)
            place['info'] = stringified_data
        return places
    
    def clean_kinds(self, original_string):
        # Splitting the string into a list of words/phrases
        words = original_string.split(',')

        # Replacing underscores with spaces and capitalizing each word
        formatted_words = [word.replace('_', ' ').capitalize() for word in words]

        # Joining the words back into a single string
        presentable_string = ', '.join(formatted_words)

        return presentable_string
    
    def stringify_dictionary(self, data):
        """
        Converts a dictionary into a readable string format. If the value is a list, it will be
        converted to a string representation of the list.
        """
        stringified_data = []
        for key, value in data.items():
            if isinstance(value, list):
                value = ', '.join(value)
            stringified_data.append(f"{key.replace('_', ' ').capitalize()}: {value}")
        return "\n".join(stringified_data)

    def get_advice(self, current_recommended_places, weatherResponse):
        prompt = f"""
        Based on the weather response and the information of the recommended places, give detailed advice to the vistors of at most 3 places they should visit and provide explanation. Provide the reasonings with the current weather.
        Label them appropriately, and go to the next line for each detail.
        This is the weather response of the area from weatherapi.com: {weatherResponse}.
        This is the information of the recommended places: {current_recommended_places}.
        """
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            temperature=0.1,
            messages=
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        info = response.choices[0].message.content.strip()
        return info