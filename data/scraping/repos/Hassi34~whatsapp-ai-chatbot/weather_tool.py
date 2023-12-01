from langchain.tools import BaseTool
from datetime import datetime
import requests
import os
from dotenv import load_dotenv

load_dotenv()
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")

desc = (
    "use this tool when you need to get any information about the weather "
    "It can provide the information about the pressure, temperature, rain/clouds, wind, sunrise, sunset and local time. "
    "It takes CITY as a parameter and returns the weather data for that specific city/location"
    "['city]."
)

class WeatherTool(BaseTool):
    name = "Weather Status Tool"
    description = desc
    def _run(self, city: str):
        try:
            CITY = city['title']  
        except TypeError:
            CITY = city

        if CITY is not None and isinstance(CITY, str):
            BASE_URL = "http://api.openweathermap.org/data/2.5/weather?"
            URL = BASE_URL + "appid=" + OPENWEATHER_API_KEY + "&q=" + CITY + "&units=metric"

            response = requests.get(URL).json()
            weather_list = []
            temp_celsius = str(response['main']['temp']) + " °C"
            wind_speed = str(response['wind']['speed']) + " km/h"
            humidity = str(response['main']['humidity']) + " %"
            air_pressure = str(response['main']['pressure']) + " hPa"
            clouds_coverage = str(response['clouds']['all']) + " %"
            description = response['weather'][0]['description']
            current_local_time = datetime.utcfromtimestamp(response['dt'] + response['timezone'])
            sunrise_time = datetime.utcfromtimestamp(response['sys']['sunrise'] + response['timezone'])
            sunset_time = datetime.utcfromtimestamp(response['sys']['sunrise'] + response['timezone'])
            
            weather_list.append({
                'temp celsius' : temp_celsius,
                'wind speed' : wind_speed,
                'humidity' : humidity,
                'air pressure' : air_pressure,
                'clouds coverage' : clouds_coverage,
                'description' : description,
                'local time' : current_local_time,
                'sunrise time' : sunrise_time,
                'sunset time' : sunset_time
            })
            return weather_list
        else:
            return "Please provide a valid name of the city"
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")
