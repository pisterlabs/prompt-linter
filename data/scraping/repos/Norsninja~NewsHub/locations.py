import os
import pickle
from datetime import datetime
import openai
import time
import json
from tqdm import tqdm
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from geopy.exc import GeocoderServiceError
import time


# Import the configuration loader
from configparser import ConfigParser

config = ConfigParser()
config.read('modules/suite_config.ini')

CACHE_FILE = config['Cache']['LocCacheFile']

# OpenAI API key
openai_api_key = config['OPENAI']['OPENAI_API_KEY']

def extract_locations(summaries):
    print("inside extract_locations")
        # Attempt to load cache
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'rb') as f:
            cache_time, locations = pickle.load(f)
        
        # If the cache is less than an hour old - return the cached data
        if (datetime.now() - cache_time).total_seconds() < 21600: # 6 hours
            return locations
    locations = []
    for summary in tqdm(summaries, desc="Extracting locations"):
        title, category, text, link, timestamp, source = summary
        print(text)
        for _ in range(3):  # Try the API call up to 3 times
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-1106",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a deterministic AI specializing in Named Entity Recognition, employed by a NewsPlanetAI, a reputable news source. You have been given the task of reading news articles and identifying one location that the news article is most likely about. Your response will be used to geocode the locations of the articles on a map. Give one location ONLY in English, in this format \"City, Country\". If the article does not provide a location, respond \"None\"."
                        },
                        {
                            "role": "user",
                            "content": f"Please give one location for this article per the instructions,  \"{text}\""
                        },
                    ],
                    temperature=0,
                    max_tokens=80,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    request_timeout=15  # Set the request timeout to 15 seconds
                )
                # If the API call succeeds, exit the loop
                
                location = response['choices'][0]['message']['content'].strip()  # remove leading/trailing spaces
                print(location)
                if location.lower() == "none":  # Log if None location
                    print(f"Headline: {title} - Location: {location} (none)")
    
                locations.append(location)

                break
            except openai.error.APIError as e:
                print(f"OpenAI API returned an API Error: {e}. Retrying...")
                time.sleep(1)  # Wait for 2 seconds before retrying
            except openai.error.APIConnectionError as e:
                print(f"Failed to connect to OpenAI API: {e}. Retrying...")
                time.sleep(1)
            except openai.error.RateLimitError as e:
                print(f"OpenAI API request exceeded rate limit: {e}. Retrying after a longer delay...")
                time.sleep(2)  # Wait longer if rate limit has been exceeded
            except openai.error.ServiceUnavailableError as e:
                print(f"OpenAI API service unavailable: {e}. Retrying...")
                time.sleep(2)  # Wait for 10 seconds before retrying
            except openai.error.Timeout as e:
                print(f"Request timed out: {e}. Retrying...")
                time.sleep(3)  # You may want to implement exponential backoff here                
        else:
            # If the API call failed 3 times, add a None location and continue with the next summary
            print("Failed to get location for a summary after 3 attempts. Skipping...")
            locations.append(None)
            continue

    # Check if the lengths of summaries and extracted_locations are different
    if len(summaries) != len(locations):
        # If they are different, append "None" to extracted_locations to match the lengths
        locations += [None] * (len(summaries) - len(locations))

    # Cache data and return locations
    cache_data = (datetime.now(), locations)
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache_data, f)
    return locations


def get_coordinates(locations):
    print("Inside get_coordinates")
    geolocator = Nominatim(user_agent="NewsPlanetAi", timeout=3)  # 10 seconds of timeout
    coordinates = []
    retries = 3  # number of retries
    delay = 5  # delay in seconds

    for location in tqdm(locations, desc="Processing locations"):
        for i in range(retries):
            try:
                # If location is "None", append None coordinates and continue
                if location == "None":
                    coordinates.append((None, None))
                    break

                # Attempt to geocode the location
                geolocation = geolocator.geocode(location)

                # If geocoding is successful, append the coordinates
                if geolocation is not None:
                    coordinates.append((geolocation.latitude, geolocation.longitude))
                    break
                else:
                    # If geocoding fails, append None coordinates
                    coordinates.append((None, None))
                    break

            except (GeocoderTimedOut, GeocoderServiceError):
                if i < retries - 1:  # i is zero indexed
                    time.sleep(delay)  # wait before trying to fetch the data again
                else:
                    print(f"Geocoding failed for location: {location}. Appending None coordinates.")
                    coordinates.append((None, None))
                    break

    return coordinates


def append_locations_to_news_json(news, summaries, locations, coordinates):
    # Iterate over the categories in the news
    for category in news['categories']:
        # Iterate over the summaries in each category
        for news_summary in category['summaries']:
            # Find the index of the summary in summaries that matches the news_summary
            indices = [i for i, summary in enumerate(summaries) if summary[0] == news_summary['headline']]
            if indices:
                index = indices[0]
                # Add the location and coordinates to the news summary
                news_summary['location'] = locations[index]
                news_summary['coordinates'] = coordinates[index]  # Add this line
    return news

def generate_geojson(summaries, extracted_locations, coordinates):
    features = []

    for i in range(len(summaries)):
        headline, category, text, url, timestamp, source = summaries[i]
        location = extracted_locations[i] if i < len(extracted_locations) else None
        coords = coordinates[i] if i < len(coordinates) else None

        feature = {
            "type": "Feature",
            "properties": {
                "headline": headline,
                "link": url,
                "text": text  # added the 'text' to the properties
            },
            "geometry": {
                "type": "Point",
                "coordinates": [coords[1], coords[0]] if coords else [None, None]
            }
        }
        features.append(feature)

    geojson_data = {
        "type": "FeatureCollection",
        "features": features
    }

    # Save the GeoJSON data to a file
    folder_path = 'geojson_data'
    os.makedirs(folder_path, exist_ok=True)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f'{folder_path}/modular_geojson_{current_time}.json'

    with open(file_name, 'w') as f:
        json.dump(geojson_data, f)

    return geojson_data, file_name


    # # Save the updated news data back to the news.json file
    # with open('news.json', 'w', encoding='utf-8') as f:
    #     json.dump(news, f, indent=4)

