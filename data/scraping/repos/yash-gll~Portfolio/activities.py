import spacy
import time
import requests
import re
import openai

# Set up the OpenAI API
openai.api_key = "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"  # Replace with your actual API key
GPT_API_URL = "https://api.openai.com/v1/chat/completions"

# Define the system message
system_msg = 'You are a classification model tasked with identifying types of places from user prompts. Match these places to specific categories based on a predefined list of parameters.'

params = '''
car_dealer
car_rental
car_repair
car_wash
electric_vehicle_charging_station
gas_station
parking
rest_stop
farm
art_gallery
museum
performing_arts_theater
library
preschool
primary_school	school
secondary_school
university
amusement_center
amusement_park
aquarium
banquet_hall
bowling_alley
casino
community_center
convention_center
cultural_center
dog_park
event_venue
hiking_area
historical_landmark
marina
movie_rental
movie_theater
national_park
night_club
park
tourist_attraction
visitor_center
wedding_venue
zoo
accounting
atm
bank
administrative_area_level_1
administrative_area_level_2
country	locality
postal_code
school_district
city_hall
courthouse
embassy
fire_station	
local_government_office
police
post_office
dental_clinic
dentist
doctor
drugstore
hospital	
medical_lab
pharmacy
physiotherapist
spa
bed_and_breakfast
campground
camping_cabin
cottage
extended_stay_hotel
farmstay
guest_house	hostel
hotel
lodging
motel
private_guest_room
resort_hotel
rv_park
church
hindu_temple
mosque
synagogue
barber_shop
beauty_salon
cemetery
child_care_agency
consultant
courier_service
electrician
florist
funeral_home
hair_care
hair_salon
insurance_agency	
laundry
lawyer
locksmith
moving_company
painter
plumber
real_estate_agency
roofing_contractor
storage
tailor
telecommunications_service_provider
travel_agency
veterinary_care
auto_parts_store
bicycle_store
book_store
cell_phone_store
clothing_store
convenience_store
department_store
discount_store
electronics_store
furniture_store
gift_shop
grocery_store
hardware_store
home_goods_store	
home_improvement_store
jewelry_store
liquor_store
market
pet_store
shoe_store
shopping_mall
sporting_goods_store
store
supermarket
wholesaler
athletic_field
fitness_center
golf_course
gym
playground
ski_resort
sports_club
sports_complex
stadium
swimming_pool
airport
bus_station
bus_stop
ferry_terminal
heliport
light_rail_station
park_and_ride	
subway_station
taxi_stand
train_station
transit_depot
transit_station
truck_stop
'''

params = params.split("\n")[1:-1]
task = f'''Given a list of allowed parameters {params} you will receive prompts describing places a user wishes to visit. Your task is to classify these prompts into corresponding categories from the list of parameters. 

Rules for the task:
1. Classifications must match the provided parameters exactly.
2. If a prompt contains multiple places, list each matching parameter separately.
3. Ensure that all classifications are relevant and accurately reflect the user's intent.
4. The response should be in the format of a list of strings, each string being a parameter that matches the place type in the prompt.
5. Do not include any categories not explicitly mentioned in the user's prompt.
6. Ensure the response is concise and free of unnecessary content or formatting.

Example Prompt: "I would like to visit some museums and art exhibitions."
Expected Output: ['museum', 'art_gallery']
'''

# Define the conversation history
messages = [
    {'role': 'system', 'content': system_msg},
    {'role': 'user', 'content': task},
]


def classify_prompt(prompt, messages = messages):
  def classify_prompt_rec(prompt, messages = messages, retries = 0, max_retries = 10):
    try:
      # Introduce new variables to prevent modification of the params, since this
      # function is recursive in the event of error.
      formatted_prompt = "Prompt: " + prompt
      formatted_messages = messages.copy()
      formatted_messages.append({'role': 'user', 'content': formatted_prompt})

      response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=formatted_messages,
        stop=None,
        temperature=0)

      re = response['choices'][0]['message']['content']
      return re

    except openai.error.RateLimitError as e:
      retry_time = e.retry_after if hasattr(e, 'retry_after') else 30
      print(f"Rate limit exceeded. Retrying in {retry_time} seconds...")
      time.sleep(retry_time)
      if retries >= max_retries:
        return None
      else:
        return classify_prompt_rec(messages, prompt, retries + 1)

    except openai.error.APIError as e:
      retry_time = e.retry_after if hasattr(e, 'retry_after') else 30
      print(f"API error occurred. Retrying in {retry_time} seconds...")
      time.sleep(retry_time)
      if retries >= max_retries:
        return None
      else:
        return classify_prompt_rec(messages, prompt, retries + 1)

    except OSError as e:
      retry_time = 5  # Adjust the retry time as needed
      print(f"Connection error occurred: {e}. Retrying in {retry_time} seconds...")
      time.sleep(retry_time)
      if retries >= max_retries:
        return None
      else:
        return classify_prompt_rec(messages, prompt, retries + 1)

  print(prompt, type(prompt))
  return classify_prompt_rec(prompt, messages=messages)


def find_nearby_attractions(types, latitude, longitude, radius=10000):
    # Google Places API endpoint
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

    # Parameters for the API request
    target = []
    word = ""
    for t in types:
      if t.isalpha() or t == "_":
        word += t
      else:
        target.append(word)
        word = ""

    target = [t for t in target if len(t)>1]
    print(target, type(target))
    params = {
        "location": f"{latitude},{longitude}",
        "radius": radius,
        "type": target,
        "key": "AIzaSyAdEd1IBe15oviCLW6QXgHu-KGu3Tqk3x0"
    }

    attractions = []
    seen_places = set()
    while True:
      response = requests.get(url, params=params)
      if response.status_code == 200:
        data = response.json()
        for place in data.get("results", []):
          if place['place_id'] not in seen_places:
            attractions.append(place)
            seen_places.add(place['place_id'])

        # Check if there's a next page
        page_token = data.get("next_page_token")
        if page_token and len(attractions) < 10:
            params["pagetoken"] = page_token
            # A short delay is required before the next page token becomes valid
            time.sleep(2)
        else:
            break

    # Sort attractions by rating and return top 10
    top_attractions = sorted(attractions, key=lambda x: x.get('rating', 0), reverse=True)[:10]
    return top_attractions

def get_details(places):
    spots = []
    for idx, place in enumerate(places):
        photos = place.get('photos', [])
        image_url = None
        if photos:
            first_photo = photos[0]
            photo_reference = first_photo.get('photo_reference', None)
            if photo_reference:
                api_key = "AIzaSyAdEd1IBe15oviCLW6QXgHu-KGu3Tqk3x0"
                max_width = 290
                image_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth={max_width}&photoreference={photo_reference}&key={api_key}"

        new_place = {
            'Image': image_url,
            'Name': place['name'],
            'Rating': place.get('rating', 'No Rating'),
            'Total_Rating': place.get('user_ratings_total', 'No Reviews'),
            'Location': place.get('vicinity', 'No Address Provided'),
        }
        spots.append(new_place)
    return spots

def geocode_location(location):
    api_key = "AIzaSyAdEd1IBe15oviCLW6QXgHu-KGu3Tqk3x0"  # Make sure to set your API key in your environment variables
    url = "https://maps.googleapis.com/maps/api/geocode/json"

    params = {
        "address": location,
        "key": api_key
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        results = response.json()["results"]
        if results:
            geometry = results[0]["geometry"]["location"]
            return geometry["lat"], geometry["lng"]  # returns a dict with 'lat' and 'lng' keys
        else:
            raise ValueError("No results found for the specified location.")
    else:
        raise ConnectionError(f"Failed to fetch data: {response.status_code}, {response.reason}")