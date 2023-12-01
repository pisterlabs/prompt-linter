import streamlit as st
import openai
import pandas as pd
import time
import plotly.express as px
from PIL import Image
# Set the OpenAI API Key (Replace 'XXXXXXXXXXXXX' with your actual API key)
openai.api_key = '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'

# Create a Streamlit app with improved design
st.set_page_config(page_title="Atlas Analytics", layout="wide")
# Custom CSS to style the app
st.markdown(
    """
    <style>
    .stButton > button {
        background-color: #4CAF50;  /* Use green color for the buttons */
        color: white;
        font-weight: bold;
        border: none;
        padding: 15px 30px;  /* Larger button */
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 18px;
        border-radius: 5px;
        cursor: pointer;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .top-banner {
        border-style: solid;
        border-color: green;
        padding: 20px;
        text-align: center;
    }
    .banner-title {
        color: white;  /* Use black for title text color */
        font-size: 40px;
    }
    .banner-description {
        color: white;  /* Use black for description text color */
        font-size: 16px;
    }
    .stPlotly {
        background-color: #4CAF50;  /* Use green for Plotly chart background */
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Business Idea Generator
def get_business_idea(country, countryOfResidence):
    with st.spinner("AI is thinking..."):
        time.sleep(3)

    prompt = f"Generate a business idea for people from {country} living in {countryOfResidence}, please provide three examples as to why they are great business ideas."
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        temperature=0.7,
        max_tokens=100
    )

    idea = response.choices[0].text.strip()
    return idea

# Load the dataset
file_path = 'Datasets/Permanent Resident - Country of Citizenship.xlsx'
df = pd.read_excel(file_path)

# Clean the data
df = df.dropna().reset_index(drop=True)
df = df[['Unnamed: 1', 'Unnamed: 3', 'Unnamed: 4']]
df.columns = ['Country', 'Percentage', 'Base Count']
df = df.iloc[1:]
df['Percentage'] = pd.to_numeric(df['Percentage'], errors='coerce')

# Exclude the total count row and get the specific country with the max percentage
specific_country_df = df.iloc[1:]
specific_country_max_percentage = specific_country_df.loc[specific_country_df['Percentage'].idxmax()]

# Top Banner
st.markdown(
    """
    <div class="top-banner">
        <h1 class="banner-title">Atlas Analytics</h1>
        <p class="banner-description">Empowering Entrepreneurs with Data Analytics</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Business Idea Generator Section
st.title("AI-Powered Business Idea Generator")
st.write("Welcome to the AI-powered Business Idea Generator. Our AI model has analyzed data from various countries and industries to provide you with unique business ideas.")
st.write("Simply select a country, and we'll use the power of AI to suggest business ideas tailored to people from that country living in your chosen location.")

# Sidebar for user input
st.sidebar.header("Select Options")
country = st.sidebar.selectbox("Select Country", df['Country'])
countryOfResidence = st.sidebar.text_input("Enter Country of Residence", "Canada")

with st.sidebar:
    image = Image.open("Orange Digital Code Logo Template.png")
    st.image(image, caption='Powered by Environics Analytics & OpenAI')
    st.write("About Us: Atlas Analytica is a dynamic analytics firm that empowers entrepreneurs with AI and data analytics. We provide innovative solutions to help businesses make data-driven decisions and achieve their goals. With us, entrepreneurs have a trusted partner to navigate the modern business landscape and seize opportunities. ")
    st.write("Coders: Jacob Manangan & Igo Domingo")

# Main content area for Business Idea Generator
st.write("Country with max percentage:", specific_country_max_percentage['Country'])
st.write("Percentage:", specific_country_max_percentage['Percentage'])
st.write("Base Count:", specific_country_max_percentage['Base Count'])

# Button to generate the business idea
if st.button("Generate Business Idea", key="business_idea_button"):
    idea = get_business_idea(country, countryOfResidence)
    st.write("Business idea for people from", country, "living in", countryOfResidence, ":")
    st.write(idea)


# Data Visualization Section
st.title("Data Visualizations")
st.write("Explore intriguing visualizations of permanent and temporary residents' data in vibrant colors and engaging charts.")
st.write("Use the following buttons to navigate the data visualizations:")

# Button for Data Visualization - Permanent Residents (Section 1)
show_permanent_residents_1 = st.button("Section 1: Permanent Residents by Country", key="permanent_residents_1_button")
if show_permanent_residents_1:
    st.markdown("### Section 1: Permanent Residents by Country")
    st.write("In this chart, you can explore the population density via ethnicity in permanent residents by country.")
    st.write("Compare the 'Count' and 'Base Count' variables to gain insights into the diverse ethnicities in your chosen region.")
    st.write("Hover over data points for more information.")
    
    PRc = pd.read_excel("Datasets/2Permanent_Resident_-_Country_of_Citizenship copy.xlsx")
    PRc = PRc.sort_values (by='Count', ascending=True)
    fig = px.bar(PRc, x='Count', y='Country', hover_data=[PRc.columns[0], 'Count', 'Base Count'])
    fig.update_layout(height=300, width=500)
    st.plotly_chart(fig, use_container_width=True)
    
    if st.button("Close Section 1", key="close_permanent_residents_1"):
        show_permanent_residents_1 = False

# Button for Data Visualization - Permanent Residents (Section 2)
show_permanent_residents_2 = st.button("Section 2: Permanent Residents by Region", key="permanent_residents_2_button")
if show_permanent_residents_2:
    st.markdown("### Section 2: Permanent Residents by Region")
    st.write("Explore the population density via ethnicity in permanent residents by region.")
    st.write("Gain an understanding of how cultural diversity shapes your selected area.")
    st.write("Hover over data points for more information.")
    
    PRr = pd.read_excel("Datasets/2Permanent_Resident_-_Region_of_Citizenship.xlsx")
    fig = px.bar(PRr, x='Count', y='Region', hover_data=[PRr.columns[0], 'Count', 'Base Count'])
    fig.update_layout(height=300, width=500)
    st.plotly_chart(fig, use_container_width=True)
    
    if st.button("Close Section 2", key="close_permanent_residents_2"):
        show_permanent_residents_2 = False

# Button for Data Visualization - Temporary Residents (Section 1)
show_temporary_residents_1 = st.button("Section 3: Temporary Residents by Region", key="temporary_residents_1_button")
if show_temporary_residents_1:
    st.markdown("### Section 3: Temporary Residents by Region")
    st.write("Explore the population density via ethnicity in temporary residents by region.")
    st.write("Gain insights into the transient population's ethnicity distribution.")
    st.write("Hover over data points for more information.")
    
    TRr = pd.read_excel("Datasets/2Temporary_Resident-_Region_of_Citizenship.xlsx")
    fig = px.bar(TRr, x='Count', y='Region', hover_data=[TRr.columns[0], 'Count', 'Base Count'])
    fig.update_layout(height=300, width=500)
    st.plotly_chart(fig, use_container_width=True)
    
    if st.button("Close Section 3", key="close_temporary_residents_1"):
        show_temporary_residents_1 = False

# Button for Data Visualization - Temporary Residents (Section 2)
show_temporary_residents_2 = st.button("Section 4: Temporary Residents by Country", key="temporary_residents_2_button")
if show_temporary_residents_2:
    st.markdown("### Section 4: Temporary Residents by Country")
    st.write("Discover the population density via ethnicity in temporary residents by country.")
    st.write("Learn more about the mix of ethnicities in the region of your choice.")
    st.write("Hover over data points for more information.")
    
    TRc = pd.read_excel("Datasets/2Temporary_Resident_-_Country_of_Citizenship.xlsx")
    TRc = TRc.sort_values( by='Count', ascending=True)
    fig = px.bar(TRc, x='Count', y='Country', hover_data=[TRc.columns[0], 'Count', 'Base Count'])
    fig.update_layout(height=300, width=500)
    st.plotly_chart(fig, use_container_width=True)
    
    if st.button("Close Section 4", key="close_temporary_residents_2"):
        show_temporary_residents_2 = False
