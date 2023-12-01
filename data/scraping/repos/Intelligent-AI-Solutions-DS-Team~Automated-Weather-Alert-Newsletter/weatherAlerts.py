import streamlit as st
import pandas as pd
import overpy
import requests
from newspaper import Article
import openai
from helpers.get_links import get_rappler_links
from streamlit_main import generate_summary

# api_key = st.sidebar.text_input("API Key", type="password", key="api_key")
openai.api_base = "https://api.openai.com/v1"
api_key = st.secrets["openai_api_key"]
openai.api_key = api_key
model = "gpt-3.5-turbo"

@st.cache_data
def get_cities_data():
    api = overpy.Overpass()

    query = """
        area[name="Philippines"][admin_level=2];
        node(area)[place=city];
        out center;
    """

    result = api.query(query)
    cities_data = []
    for node in result.nodes:
        city_name = node.tags.get("name", "Unknown")
        latitude = node.lat
        longitude = node.lon
        city_info = {
            "city_name": city_name,
            "latitude": latitude,
            "longitude": longitude
        }
        cities_data.append(city_info)
    return cities_data

cities_data = get_cities_data()

def load_data(sheets_url):
    csv_url = sheets_url.replace("/edit", "/export?format=csv")
    return pd.read_csv(csv_url)

gsheets_url = st.secrets["public_gsheets_url"]


def get_coordinates(city_name):
    for city_data in cities_data:
        if city_data['city_name'] == city_name:
            return city_data['latitude'], city_data['longitude']
    return None, None  # Return None if city name not found in the list

def fetch_weather_alerts(lat, lon):
        url = f"https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&exclude=hourly,minutely,daily&appid=63cab36178a3d987d900af49583e506d"
        response = requests.get(url)
        data = response.json()
        return data.get('alerts', [])
    
def determine_status_and_description(alerts):
    if alerts:
        return alerts[0].get('event'), alerts[0].get('description')
    else:
        return 'No Alert', 'No Description'
    
st.title('⚠️ Weather Alerts')
st.markdown(f"[**Employee Database**]({gsheets_url})")
fetch_alerts = st.button('Fetch Weather Alerts', key="fetch_weather_alerts")
st.info("This app fetches weather alerts displays them in a table.")

if fetch_alerts:
    if api_key:
        df = load_data(gsheets_url)
        with st.spinner("Fetching Weather Alerts..."):
            df['Latitude'], df['Longitude'] = zip(*df['city'].map(get_coordinates))
            df['status'], df['description'] = zip(*df.apply(lambda row: determine_status_and_description(fetch_weather_alerts(row['Latitude'], row['Longitude'])), axis=1))
            # sort df alphabetically by status column
            df = df.sort_values(by=['status'])
            st.dataframe(df, hide_index=True)
            grouped_df = df.groupby('status')
            for status, group in grouped_df:
                if status != 'No Alert':
                    st.markdown(f"### {status}")
                    for index, row in group.iterrows():
                        st.markdown(f"**{row['name']}** - {row['city']}")

    else:
        st.error("Please enter your API key.")