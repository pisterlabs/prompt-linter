# 필요한 라이브러리 임포트
import datetime
import streamlit as st
import openai
import requests

# st.secrets로부터 API 키 가져오기
openai.api_key = st.secrets["OPENAI_API_KEY"]
weather_api_key = st.secrets["OPENWEATHER_API_KEY"]

# Streamlit 앱의 제목 설정
st.title("Weather-appropriate Outfit Image Generator")

# 사용자 사진 업로드
uploaded_file = st.file_uploader("Upload a photo of yourself", type=["jpg", "jpeg", "png"])
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

# 도시 목록
cities = ["Seoul", "Suwon", "Incheon", "Daejeon", "Busan", "Gangneung", "Jeju", "Cheongju", "Chungju", "Gwangju", "Daegu", "Jeonju", "Yeosu", "Mokpo", "Ulsan", "Pohang", "Changwon"]

# 도시 이름을 드롭다운 목록에서 선택
selected_city = st.selectbox("Select a city:", cities)

# 날짜 선택
selected_date = st.date_input("Select a date for weather information", datetime.date.today())

# 성별 선택
gender = st.selectbox("Select your gender:", ["남자", "여자"])

# 사용자가 원하는 옷 스타일 선택
outfit_styles = ["Casual", "Formal", "Sporty", "Business Casual", "Traditional", "Dress", "Sportswear", "Outdoor", "T-shirt", "V-neck T-shirt", "Long-sleeved T-shirt", "Polo shirt", "Henry neck shirt", "Sleeveless shirt", "Sweater", "Hoodie", "Dress shirt", "Blouse", "Coat", "Jacket", "Cardigan", "Padding", "Fleece", "Hood zip-up", "Bomber jacket", "Parka", "Jumper jacket", "Knit vest", "Skirt", "Long skirt", "Suit skirt", "Pants", "Suspender skirt", "Jean skirt", "Jeans", "Slacks", "Cotton pants", "Shorts"]
selected_style = st.selectbox("Select your preferred style:", outfit_styles)

# OpenWeather API를 사용하여 도시 이름을 기반으로 날씨 데이터 가져오기
def get_weather_data_by_city(city_name, date, api_key=weather_api_key):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        temp = round(data['main']['temp'] - 273.15, 1)  # Convert Kelvin to Celsius and round to one decimal
        humidity = data['main']['humidity']
        wind_speed = data['wind']['speed']
        month = date.month
        if 3 <= month <= 5:
            season = "봄"
        elif 6 <= month <= 8:
            season = "여름"
        elif 9 <= month <= 11:
            season = "가을"
        else:
            season = "겨울"
        return temp, humidity, wind_speed, season
    else:
        st.write("Error fetching weather data.")
        return None, None, None, None

# 선택된 도시의 날씨 정보 출력
temp, humidity, wind_speed, season = get_weather_data_by_city(selected_city, selected_date)
if temp and humidity and wind_speed and season:
    st.write(f"Temperature: {temp}°C")
    st.write(f"Humidity: {humidity}%")
    st.write(f"Wind Speed: {wind_speed} m/s")
    st.write(f"Season: {season}")

# OpenAI GPT-3를 사용하여 선택된 스타일, 성별, 계절, 기온 및 풍속을 기반으로 옷에 대한 설명 생성
if st.button("Describe Outfit"):
    response = openai.Completion.create(
      engine="text-davinci-002",
      prompt=f"Describe a {selected_style} outfit suitable for a {gender} during {season}, considering the temperature is {temp}°C and wind speed is {wind_speed} m/s.",
      max_tokens=100
    )
    description = response.choices[0].text.strip()
    st.write(description)
    
    # DALL·E를 사용하여 이미지 생성
    with st.spinner("Generating Image..."):
        dalle_response = openai.Image.create(
            prompt=description,
            size="512x512"
        )
    # 생성된 이미지 표시
    st.image(dalle_response["data"][0]["url"])
