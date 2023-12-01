from datetime import datetime

import json
import openai
import re
import utils

def getTop5RecommendedHotels(hotels):
    if not hotels or 'start_date' not in hotels[0] or 'end_date' not in hotels[0]:
      return {"status": 400, "error": "Failed to fetch or process hotels"}

    start_date_of_first_hotel = hotels[0]['start_date']
    end_date_of_first_hotel = hotels[0]['end_date']

    hotel_list_text = utils.createHotelString(hotels)
    #print("HOTEL TEXT")
    #print(hotel_list_text)
    top5Hotels = getTopRecommendedHotelFromGPT(hotel_list_text, start_date_of_first_hotel, end_date_of_first_hotel)

    return top5Hotels


def parse_uuids_from_gpt_response(response):
    return re.findall(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', response)



def getTopRecommendedHotelFromGPT(hotels_str, start_date, end_date):
  today_date = datetime.today().strftime('%Y-%m-%d')
  completion = openai.ChatCompletion.create(
    model="gpt-4-0613",
    temperature=0,
    messages=[
      {"role": "system", "content": "From the following list of hotels, return a sorted array with the IDs of the top 5 hotels that you believe (based on what you already know from those hotels and the list I am sharing here, with the most relevant one in the first position) are the preffered by Costa Ricans to stay between the dates " + start_date + " and " + end_date + ". Today is " + today_date + ". The price is in USD. The format must be [id1,id2,id3,id4,id5]. Here is the list of hotels: " + hotels_str}
    ]
  )
  
  content = completion.choices[0].message['content']

  print("GPT RESPONSE")
  print(content)
  
  uuid_list = parse_uuids_from_gpt_response(content)

  return uuid_list