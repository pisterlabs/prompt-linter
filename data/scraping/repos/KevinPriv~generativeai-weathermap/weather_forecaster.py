from flask import Flask, request, render_template_string, render_template
from concurrent.futures import ThreadPoolExecutor
from flask_executor import Executor
import requests
import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import openai
import folium

load_dotenv(Path(".env"))

app = Flask(__name__, template_folder='templateFiles', static_folder='staticFiles')
executor = Executor(app)

weather_api_key = os.getenv("WEATHERSTACK_API")
openai_api_key = os.getenv("OPENAI_API")


@app.route('/', methods=['GET', 'POST'])
def map_display():
    map_html = ''

    if request.method == 'POST':
        location = request.form.get('location')
        r = requests.get(f'http://api.weatherstack.com/current?access_key={weather_api_key}&query={location}')
        weather_json = r.json()
        if not weather_json.get('success', True):
            return render_template('map.html', map_html="Could not find place.")

        latitude = weather_json['location']['lat']
        longitude = weather_json['location']['lon']
        location_name = weather_json["location"]["name"]
        temperature = weather_json["current"]["temperature"]
        weather_desc = weather_json["current"]["weather_descriptions"][0]
        wind_speed = weather_json["current"]["wind_speed"]
        humidity = weather_json["current"]["humidity"]

        future = executor.submit(generate_image, location_name, weather_desc)
        image = future.result()

        m = folium.Map(location=[latitude, longitude], zoom_start=11)

        css_output_string = "<style> img { vertical-align: text-top; } </style>"
        output_string = f"{css_output_string}<img src=\"{image}\" height=256 width=256></img><b>Location</b>: {location_name}<br><b>Temperature</b>: {temperature} degrees<br><b>Weather</b>: {weather_desc}<br><b>Wind Speed</b>: {wind_speed} km/h<br><b>Humidity</b>: {humidity}%"

        folium.Marker(location=[latitude, longitude],
                      popup=output_string,
                      icon=folium.Icon(color="red")).add_to(m)

        map_html = m._repr_html_()

    return render_template('map.html', map_html=map_html)


def generate_image(city_name, weather):
    openai.api_key = os.getenv("OPENAI_KEY")

    response = openai.Image.create(
        prompt="A " + weather + " day in " + city_name + ".",
        n=1,
        size="256x256",
    )

    return response["data"][0]["url"]


app.run(debug=True)
