import streamlit as st
import requests
from openai import OpenAI
import json

def getweather_today(lat,lon):
    '''Get Weather for Today'''
    method="weather"
    url = f'{api_url}/{method}?lat={lat}&lon={lon}&appid={weather_key}'
    response = requests.get(url)

    data = response.json()
    return data

def getweather_next7days(lat,lon):
    '''Get Weather for Next 7 Days'''
    method="forecast"
    url = f'{api_url}/{method}?lat={lat}&lon={lon}&cnt=40&appid={weather_key}'
    response = requests.get(url)

    data = response.json()
    return data

# Function to fetch data from an API
def fetch_data(city, country_code, api_key):
    # Example URL - replace with the actual API you are using
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},{country_code}&appid={api_key}"
    response = requests.get(url)
    return response.json()

def getlatlon(city, countrycode):
    '''Get Latitude & Longitude'''
    prompt_message = f"Give me the latitude and longitude for {city}, {countrycode} with two digit after the point in json format"
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt_message,
            }
        ],
        model="gpt-3.5-turbo",
    )

    return response.choices[0].message.content

def Generate_desc(raw_json_data:str) -> dict:

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant, when the user gives you a weather data, you "
                           "should tell them what to wear and what to do based on the weather data they provided for the next 5 days"
                           "the data user provided will be in json and the temperature is in kelvin "
                           " The result must be in json, and it must have 3 sigment the first sigment have to be titled as 'recommendations' "
                           "followed with date(yyyy-mm-dd) and the recommended cloth to wear , and the second sigment have to be "
                           "titled as 'outfit_description', followed with date(yyyy-mm-dd) and description of what the user might looks like based on the recommended cloth. the last"
                           "sigment have to be titled as 'activities' and followed with date(yyyy-mm-dd) and activities"
                           "they should be doing it would be nice if activites recommendation would include some brand name"
            },
            {
                "role": "user",
                "content": f"{raw_json_data}",
            }

        ],
        model="gpt-3.5-turbo-1106",
        response_format={"type": "json_object"},
        temperature=0.1
    )

    data = json.loads(chat_completion.choices[0].message.content)
    return data, chat_completion

def Generate_image(prompt:str) -> str:
    response = client.images.generate(
        model="dall-e-3",
        prompt=f"""Generating character wearing the clothes and weather in the background based on this: "{prompt}" in family friendly cute pixel art""",
        size="1024x1024",
        quality="standard",
        n=1,
    )
    return response.data[0].url

# Streamlit app layout
def main():
    st.title("City Information App")

    # Input fields

    city = st.text_input("Enter city name")
    country_code = st.selectbox("Select country code", ["US", "CA", "UK", "AU", "DE", "FR", "JP"])
    five_days_data_checked = st.checkbox("For five days suggestion")
    #latitude_longtitude = json.loads(getlatlon(city, country_code))


    # Button to fetch data
    if st.button("Show Data"):
        if city and country_code:
            if not five_days_data_checked:
                data = getweather_today('-79.416', "43.7")
                output_data = Generate_desc(data)
                st.write(output_data[0])
                first_data = output_data[0]
                url = Generate_image(next(iter(first_data.items())))
                st.image(url, caption="Image from URL")
                #st.write(data)  # Display raw data as an example
                # You can add more code here to display pictures and formatted data
            else:
                data = getweather_next7days('-79.416', "43.7")
                #st.write(data)
                output_data = Generate_desc(data)
                first_data = output_data[0]
                url = Generate_image(next(iter(first_data.items())))
                st.write(output_data[0])
                st.image(url, caption="Image from URL")
        else:
            st.error("Please fill in all fields")





if __name__ == "_main_":
    main()