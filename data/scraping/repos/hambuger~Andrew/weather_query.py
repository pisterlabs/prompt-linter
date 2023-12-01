import os

import requests
from openai_util.function_call.openaifunc_decorator import openai_func


@openai_func
def get_weather(location_name: str):
    """
    Get the weather conditions for a location
    :param location_name: place name
    """
    geocode_url = f'''https://restapi.amap.com/v3/geocode/geo?address={location_name}&output=json&key={os.getenv('OPEN_WEATHER_MAP_KEY')}'''
    geocode_url_response = requests.get(geocode_url).json()
    if not geocode_url_response or not geocode_url_response['geocodes']:
        return None
    code = geocode_url_response['geocodes'][0]['adcode']

    weather_url = f'''https://restapi.amap.com/v3/weather/weatherInfo?city={code}&key={os.getenv('OPEN_WEATHER_MAP_KEY')}'''
    weather_url_response = requests.get(weather_url).json()
    if not weather_url_response or not weather_url_response['lives']:
        return None
    return weather_url_response['lives'][0]

# print(json.dumps(get_weather('hangzhou')))
