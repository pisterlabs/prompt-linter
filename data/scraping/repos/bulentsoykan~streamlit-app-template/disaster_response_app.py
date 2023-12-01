import streamlit as st
import openai
import os
import pandas as pd
import matplotlib.pyplot as plt
import random
from streamlit_folium import folium_static
from streamlit.components.v1 import html
import folium
from streamlit import components

import requests
import json
from datetime import datetime

from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Retrieve API key
openai.api_key = os.getenv("OPENAI_API_KEY")


# Function to query the LLM (GPT-3) for analysis based on multiple data sources
def query_llm(social_media_data, sensor_data, news_data):
    prompt = f"Analyze the following real-time data and provide recommendations:\nSocial Media Data: {social_media_data}\nSensor Data: {sensor_data}\nNews Data: {news_data}\n"
    response = openai.Completion.create(
        engine="text-davinci-002", prompt=prompt, max_tokens=500
    )
    return response.choices[0].text.strip()


# Enable session state
if "inventory" not in st.session_state:
    st.session_state.inventory = {
        "Medical Supplies": 0,
        "Rescue Teams": 0,
        "Vehicles": 0,
    }


from bs4 import BeautifulSoup


# Function to scrape news headlines from BBC News
def fetch_news():
    url = "https://www.bbc.com/news"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    headlines = soup.find_all(
        "h3", class_="gs-c-promo-heading__title gel-pica-bold nw-o-link-split__text"
    )
    return [headline.text for headline in headlines[:15]]  # Return top 5 headlines


# UCF banner image
st.image("./UCF.jpg", width=600, caption="University of Central Florida")


def get_road_route(start_coords, end_coords):
    url = f"http://router.project-osrm.org/route/v1/driving/{start_coords[1]},{start_coords[0]};{end_coords[1]},{end_coords[0]}?overview=full&geometries=geojson"
    response = requests.get(url)
    data = json.loads(response.content)
    if data.get("routes"):
        route_coordinates = data["routes"][0]["geometry"]["coordinates"]
        route_coordinates = [(lat, lon) for lon, lat in route_coordinates]
        return route_coordinates
    else:
        return None


# Dummy function to simulate route calculation
def calculate_route(start, end):
    # Replace with your actual routing logic
    distance = "15 miles"
    time = "20 minutes"
    traffic = "Moderate"
    return distance, time, traffic


# Custom CSS to make the interface colorful
st.markdown(
    """
    <style>
        .reportview-container {
            background: linear-gradient(to bottom, #5d54a4, #9c95b8);
        }
        .sidebar .sidebar-content {
            background: linear-gradient(to bottom, #9c95b8, #5d54a4);
        }
        h1 {
            color: black;
        }
        h2 {
            color: black;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit App
st.title("🚨 Intelligent Disaster Response 🚨")

# Sidebar for navigation
st.sidebar.title("🌐 Navigation")
page = st.sidebar.selectbox(
    "Select a page",
    [
        "🏠 Home",
        "📦 Resource Allocation",
        "📦 Resource Inventory",
        "🗺️ Routing",
        "🌡️ Sensor Data",
        "📰 News Feeds",
    ],
)


# Function to create a folium map with road-based routing
def create_map_with_route(start_coords, end_coords):
    m = folium.Map(location=start_coords, zoom_start=15)
    folium.Marker(start_coords, tooltip="Start").add_to(m)
    folium.Marker(end_coords, tooltip="End").add_to(m)

    route_coordinates = get_road_route(start_coords, end_coords)
    if route_coordinates:
        folium.PolyLine(route_coordinates, color="blue", weight=2.5, opacity=1).add_to(
            m
        )
    else:
        st.warning("Could not get road-based route.")

    return m


# Initialize session state if not already initialized
if "data" not in st.session_state:
    st.session_state["data"] = []

# Home page for real-time data analysis
if page == "🏠 Home":
    st.header("📊 Real-time Data Analysis 📊")

    # Collect mock real-time data
    social_media_data = st.text_input(
        "📱 Social media data", "Fire reported at location X"
    )
    sensor_data = st.text_input("🌡️ Sensor data", "Temperature: High, Wind: Low")
    news_data = st.text_input("📰 News data", "Wildfire spreading rapidly")

    st.header("📝 LLM Recommendations 📝")
    if st.button("Generate Recommendations"):
        llm_output = query_llm(social_media_data, sensor_data, news_data)
        st.success(f"Recommendation: {llm_output}")
        st.session_state["data"].append(
            {
                "sensor_data": sensor_data,
                "news_data": news_data,
                "social_media_data": social_media_data,
                "recommendation": llm_output,
            }
        )


# Page for resource allocation
elif page == "📦 Resource Allocation":
    st.header("📦 Resource Allocation 📦")

    # Collect mock resource data
    resources = st.text_input(
        "🚒 Enter available resources", "10 Firetrucks, 20 Ambulances"
    )

    # Simulate resource optimization
    if st.button("🔍 Optimize Resources"):
        st.success("📋 Resources optimized. Ready for deployment.")


# Page for Resource Inventory
elif page == "📦 Resource Inventory":
    st.header("📦 Resource Inventory 📦")

    # Display current inventory
    st.write("### Current Inventory")
    for resource, count in st.session_state.inventory.items():
        st.write(f"{resource}: {count}")

    # Add or update resources
    st.write("### Add or Update Resources")
    resource_type = st.selectbox(
        "Select Resource Type", ["Medical Supplies", "Rescue Teams", "Vehicles"]
    )
    resource_count = st.number_input(f"Enter number of {resource_type}", min_value=0)

    if st.button("Update Inventory"):
        st.session_state.inventory[resource_type] = resource_count
        st.success(f"Inventory updated: {resource_type} set to {resource_count}")


# Page for routing
elif page == "🗺️ Routing":
    st.header("🗺️ Routing for Rescue Operations 🗺️")

    # Set default coordinates to University of Central Florida for start
    default_start = "28.6024, -81.2001"

    # Set default coordinates to Partnership II Building for end
    default_end = "28.5895, -81.1893"

    # Collect routing data
    start_location = st.text_input(
        "🅰️ Enter start location (latitude, longitude)", default_start
    )
    end_location = st.text_input(
        "🅱️ Enter end location (latitude, longitude)", default_end
    )

    start_coords = tuple(map(float, start_location.split(",")))
    end_coords = tuple(map(float, end_location.split(",")))

    # Simulate routing algorithm
    if st.button("🔍 Plan Route"):
        distance, time, traffic = calculate_route(start_coords, end_coords)

        # Display route details
        st.subheader("Route Details")
        st.write(f"**Estimated Distance:** {distance}")
        st.write(f"**Estimated Time:** {time}")
        st.write(f"**Traffic Conditions:** {traffic}")

        st.success("📋 Optimal route planned. Ready for action.")

        # Create and display the map (Assuming create_map_with_route is your function)
        m = create_map_with_route(start_coords, end_coords)
        html_data = m._repr_html_()
        st.components.v1.html(html_data, width=800, height=400)

        # Additional Insights
        st.subheader("Additional Insights")
        st.write("1. Points of Interest: Gas stations, rest stops, etc.")
        st.write("2. Safety Tips: Weather conditions, road closures.")
        st.write("3. Alternative Routes: Shortest, fastest, scenic.")


# Page for Sensor Data
elif page == "🌡️ Sensor Data":
    st.header("🌡️ Sensor Data Integration 🌡️")

    # Collect mock sensor data
    weather_data = st.text_input(
        "🌦️ Enter weather data", "Temperature: 25°C, Humidity: 60%"
    )
    seismic_data = st.text_input("🌍 Enter seismic data", "Richter Scale: 2.5")
    other_sensor_data = st.text_input("🔬 Enter other sensor data", "Air Quality: Good")

    # Simulate data integration
    if st.button("🔍 Integrate Sensor Data"):
        st.success("📋 Sensor data integrated. Ready for analysis.")


# Page for News Feeds
elif page == "📰 News Feeds":
    st.header("📰 Real-time News Updates 📰")

    # Specify the source of the news
    st.subheader("News Source: BBC News")

    # Fetch and display news headlines
    news_headlines = fetch_news()
    fetch_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.write(f"News fetched at: {fetch_time}")
    for i, headline in enumerate(news_headlines):
        st.subheader(f"{i+1}. {headline}")
