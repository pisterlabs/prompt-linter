import requests
import geocoder
from openai import OpenAI

client = OpenAI()

temperature = 0.6
basic_model = "gpt-3.5-turbo-16k"


def geocoder_api(query):
    g = geocoder.geonames(query, key='eliaswf', maxRows=1)
    return g.lat, g.lng


def summarize(query:str, content: str) -> str:
    """
    Summarizes a piece of text using the OpenAI API.

    :param query: The query to summarize.
    :type query: str
    :param content: The text to summarize.
    :type content: str
    :return: The summary.
    :rtype: str
    """
    response = client.chat.completions.create(model=basic_model,
    messages=[{'role': 'system', 'content': f'There was a search for the following weather:\n"{query}"\nPlease '
                                            f'provide a concise summary of the following content while keeping '
                                            f'mind what will best respond to the query:\n{content}\n'}],
    max_tokens=400,
    n=1,
    stop=None,
    temperature=temperature)
    summary = response.choices[0].message.content
    return summary


def get_weather(city_name):
    api_key = '916a78d6305cef8f326831938dfe03f7'
    lat, lng = geocoder_api(city_name)
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lng}&appid={api_key}&units=imperial"

    response = requests.get(url)

    if response.status_code == 200:
        weather_data = response.json()
        formatted_response = ""

        # Extract the city information
        city = weather_data['city']['name']
        country = weather_data['city']['country']
        formatted_response += f"Weather Forecast for {city}, {country}\n\n"

        # Go through each weather entry in the list
        for entry in weather_data['list']:
            # Convert temperature from Kelvin to Celsius
            temp_farenheit = entry['main']['temp']
            feels_like_farenheit = entry['main']['feels_like']
            temp_min_farenheit = entry['main']['temp_min']
            temp_max_farenheit = entry['main']['temp_max']

            # Format the date and time
            formatted_date = entry['dt_txt']

            # Add the details to the response
            formatted_response += f"{formatted_date}\n"
            formatted_response += f"   - Temperature: {temp_farenheit:.2f}°F (Feels like: {feels_like_farenheit:.0f}°F)\n"
            formatted_response += f"   - Min Temperature: {temp_min_farenheit:.0f}°F\n"
            formatted_response += f"   - Max Temperature: {temp_max_farenheit:.0f}°F\n"
            formatted_response += f"   - Pressure: {entry['main']['pressure']} hPa\n"
            formatted_response += f"   - Humidity: {entry['main']['humidity']}%\n"
            formatted_response += f"   - Weather: {entry['weather'][0]['description'].capitalize()}\n"
            formatted_response += f"   - Cloudiness: {entry['clouds']['all']}%\n"
            formatted_response += f"   - Wind: {entry['wind']['speed']} m/s, {entry['wind']['deg']} degrees\n"
            if 'rain' in entry:
                formatted_response += f"   - Rain Volume: {entry['rain']['3h']} mm/3h\n"
            formatted_response += f"   - Probability of Precipitation: {entry['pop'] * 100}%\n\n"

        #return summarize(city_name, formatted_response)
        return formatted_response
    else:
        return "City not found or request failed"
