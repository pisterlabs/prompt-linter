import requests
import json
import datetime
import openai

# Set OpenAI API key
openai.api_key = "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"

# KakaoTalk API URL and Access Token
url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"

# KMA API URL and API key
weather_url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtFcst"
api_key = '+OkWyS+jCSbH8iy6EgVNtvfDxfKo9ImIII2zL/qdiWXOzs0u5aNFjpZ782duf46IjgD99p9VA5BmiSS8IeosQw=='

# Location information (latitude, longitude)
nx = "60"
ny = "127"

# Fine dust level API URL
dust_url = "http://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getMsrstnAcctoRltmMesureDnsty"

# Get current weather data
weather_response = requests.get(weather_url, params={
    "serviceKey": api_key,
    "numOfRows": 10,
    "pageNo": 1,
    'dataType' : 'json',
    #"base_date": datetime.datetime.today().strftime("%Y%m%d"),
    "base_date": "20230423",
    "base_time": datetime.datetime.now().strftime("%H%M"),
    "nx": nx,
    "ny": ny,
})
#print('weather_response: ', weather_response);
#print('weather_response.text: ', weather_response.text);
weather_data = json.loads(weather_response.text)
print('weather_data: ', weather_data);

# Get fine dust level data
dust_response = requests.get(dust_url, params={
    "stationName": "종로구",
    #"dataTerm": "DAILY",
    "dataTerm": "MONTH",
    "ver": "1.0",
    "serviceKey": api_key,
    "returnType": 'json',
})
#print('dust_response: ', dust_response);
#print('dust_response.text: ', dust_response.text);
dust_data = json.loads(dust_response.text)
print('dust_data: ', dust_data);

# Extract relevant weather information
weather_description = weather_data['response']['body']['items']['item'][0]['fcstValue']
current_temp = float(weather_description)
current_humidity = weather_data['response']['body']['items']['item'][1]['fcstValue']
wind_speed = weather_data['response']['body']['items']['item'][3]['fcstValue']
forecast = weather_data['response']['body']['items']['item'][4]['fcstValue']

# Use ChatGPT to generate weather forecast from a 10-year weathercaster's perspective
#prompt = f"As a 10-year weathercaster, I predict that today's weather will be {forecast}."
prompt = f"Generate a weatherman line that describes the current weather, given that today's temperature is {weather_description} degrees Celsius."
print('prompt: ', prompt);
response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=prompt,
    max_tokens=1000,
    n=1,
    stop=None,
    temperature=0.7,
)
print('openAI response: ', response);
forecast_10_year = response.choices[0].text.strip()

# Determine what clothes to wear based on current temperature
if current_temp < 5:
    clothing_recommendation = "It's very cold, wear a heavy coat, scarf, gloves, and boots."
elif current_temp < 10:
    clothing_recommendation = "It's cold, wear a warm coat, scarf, and boots."
elif current_temp < 15:
    clothing_recommendation = "It's cool, wear a light jacket or sweater."
elif current_temp < 20:
    clothing_recommendation = "It's mild, wear a light jacket or sweater."
elif current_temp < 25:
    clothing_recommendation = "It's warm, wear a t-shirt or blouse."
else:
    clothing_recommendation = "It's hot, wear light clothing."

# Determine the fine dust level
dust_level = int(dust_data['response']['body']['items'][0]['pm10Value'])

if dust_level <= 30:
    dust_recommendation = "The fine dust level is good."
elif dust_level <= 80:
    dust_recommendation = "The fine dust level is moderate. It is recommended to reduce outdoor activities and wear a mask."
elif dust_level <= 150:
    dust_recommendation = "The fine dust level is bad. It is recommended to avoid outdoor activities and wear a mask."
else:
    dust_recommendation = "The fine dust level is very bad. It is recommended to stay indoors and wear a mask."

# Construct the message to be sent by the chatbot
message = f"Good morning! As a 10-year weathercaster, I predict that today's weather will be {forecast_10_year}. The current temperature is {current_temp}°C with a humidity of {current_humidity}% and wind speed of {wind_speed}m/s. {clothing_recommendation} {dust_recommendation}"
print('message: ', message)

# 🚀 Send the message via KakaoTalk API========================================
access_token = 'AmJ4oAPUbagnUghmJGcFNq8YWsuYz1n1m7euW2nzCinJYAAAAYevM9Pq'
#============================================================================


headers = {'Authorization': f'Bearer {access_token}'}
# {
#     "object_type": "text",
#     "text": "Hello, World!",
#     "link": {
#         "web_url": "https://example.com",
#         "mobile_web_url": "https://m.example.com",
#     }
# Split the message into chunks of 200 characters or less
message_chunks = [message[i:i+200] for i in range(0, len(message), 200)]

# Send each message chunk as a separate message
for chunk in message_chunks:
    data = {'template_object': json.dumps({
        "object_type": "text",
        "text": chunk,
        "link": {
            "web_url": "https://example.com",
            "mobile_web_url": "https://m.example.com",
        }
    })}
    response = requests.post(url, headers=headers, data=data)

# Check if the message was sent successfully
if response.status_code == 200:
    print("Message sent successfully!")
else:
    print(f"Error {response.status_code}: {response.text}")