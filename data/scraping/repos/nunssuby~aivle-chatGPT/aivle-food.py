import requests
import openai
import streamlit as st

# OpenAI API 인증 정보 설정
openai.api_key = "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"

# 날씨 API URL 및 인증 정보 설정
weather_api_url = "https://api.openweathermap.org/data/2.5/weather"
weather_api_key = "YOUR_API_KEY"

# 사용자 위치 정보 입력 받기
city_name = st.text_input("현재 위치를 입력하세요: ")

# 날씨 API를 이용하여 기온과 날씨 정보 가져오기
response = requests.get(weather_api_url, params={"q": city_name, "appid": weather_api_key})
weather_data = response.json()
temperature = weather_data["main"]["temp"]
weather_desc = weather_data["weather"][0]["description"]

# 사용자의 식품 알레르기 정보, 취향, 건강 상태 등 입력 받기
allergy_info = st.text_input("어떤 식품 알레르기가 있나요?")
taste_info = st.text_input("어떤 음식을 좋아하시나요?")
health_info = st.text_input("건강 상태는 어떤가요?")

# OpenAI API를 이용하여 음식 추천 문장 생성하기
prompt = f"이번에 드실 음식은 {temperature}도의 {weather_desc} 날씨에 {allergy_info} 알레르기를 고려하여 {taste_info}을(를) 좋아하는 분께 추천하는 건강한 음식은 무엇인가요?"
generated_text = openai.Completion.create(engine="text-davinci-002", prompt=prompt, max_tokens=1024)["choices"][0]["text"]

# 생성된 문장 출력하기
st.write(generated_text)
