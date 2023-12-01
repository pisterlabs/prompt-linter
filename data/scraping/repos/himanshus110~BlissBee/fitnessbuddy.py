import requests

import openai
import json
import multiprocessing
import time
from django.conf import settings

import os
from dotenv import load_dotenv

load_dotenv()


openai.api_key = os.getenv("OPEN_API_KEY")

def get_today_activity_data():
    # Get the smartwatch data from the watch
    url = "https://v1.nocodeapi.com/tusharsh21/fit/CAjeyAJwcpudQfou/aggregatesDatasets?dataTypeName=steps_count,active_minutes,calories_expended,heart_minutes,sleep_segment,weight,activity_summary&timePeriod=today&durationTime=daily"
    params = {}
    r = requests.get(url = url, params = params)
    result = r.json()

    # Initialize a dictionary to store the filtered data
    filtered_data = {}

    # Extract and add step counts if available
    steps_count = result.get('steps_count', [])
    if steps_count:
        filtered_data['steps_count'] = steps_count[0]['value']

    # Extract and add calories expended if available
    calories_expended = result.get('calories_expended', [])
    if calories_expended:
        filtered_data['calories_expended'] = calories_expended[0]['value']

    # Extract and add active minutes if available
    active_minutes = result.get('active_minutes', [])
    if active_minutes:
        filtered_data['active_minutes'] = active_minutes[0]['value']

    # Extract and add heart minutes if available
    heart_minutes = result.get('heart_minutes', [])
    if heart_minutes:
        filtered_data['heart_minutes'] = heart_minutes[0]['value']

    return filtered_data


def new_json():
  # Create a new dictionary to store the restructured data
  url = "https://v1.nocodeapi.com/tusharsh21/fit/CAjeyAJwcpudQfou/aggregatesDatasets?dataTypeName=steps_count,active_minutes,calories_expended,heart_minutes,sleep_segment,weight,activity_summary&timePeriod=7days&durationTime=daily"
  params = {}
  r = requests.get(url = url, params = params)
  result = r.json()
  restructured_data = {}

  # Iterate through each category
  for category, data_list in result.items():
      # Iterate through each dictionary in the list
      for entry in data_list:
          # Create the 'startTime-endTime' key as the concatenation of 'startTime' and 'endTime'
          time_range = f"{entry['startTime']} - {entry['endTime']}"

          # Get the 'value' for the current entry
          value = entry['value']

          # Check if the time_range already exists in the restructured data
          if time_range in restructured_data:
              # If it exists, update the existing dictionary with the category and value
              restructured_data[time_range][category] = value
          else:
              # If it doesn't exist, create a new dictionary with the category and value
              restructured_data[time_range] = {category: value}
  return restructured_data






def health_recommendation(user_data):
    prompt = f''' Act as the world's best Health and Fitness trainer who provides the best Personalized Health Recommendations to the user using the Data provided by the user from their Smartwatch which tracks their daily steps, calories burnt,
    Daily activity and other metrics. Create personalized health recommendations based on the user's daily steps and calories expended by analyzing the data provided in input.
    The generated content should be very specific and tailored to the user. The personalized plan has to be made after analyzing the smartwatch data provided as input.
    You have to generate content using the INPUT DATA of the user. The INPUT DATA is a json file with the time period being the primary key and the metrics and activities of the user during the mentioned time period being the values for the key.
    Create a personalized plan where you consider the analysis of user data and output only your suggestions and it should be concise.

    INPUT DATA:
    {user_data}

    OUTPUT FORMAT:
    Numbered List of 5 elaborate points only
    '''

    pp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {"role": "system", "content": prompt},
        ],
        max_tokens=2500,
        temperature=0.2
        )

    plan = pp['choices'][0]['message']['content']

    with open('plan1.txt', 'w') as f:
      f.write(plan)
    
    with open('plan1.txt', 'r') as f:
      content = f.read()

    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

    # Rejoin the paragraphs to remove extra empty lines
    cleaned_text = '\n'.join(paragraphs)

    media_directory = settings.MEDIA_ROOT
    file_path = os.path.join(media_directory, 'plan1.txt')
    with open(file_path, 'w') as f:
      f.write(cleaned_text)


    
    


def exercise_recommendation(user_data):
    prompt = f''' Act as the world's best Fitness trainer and dietitian who provides the best Personalized Fitness plans and diet to the user using the Data provided by the user from their Smartwatch which tracks their daily steps, calories burnt,
    Daily activity and other metrics. Create personalized Fitness and diet plans based on the user's daily steps and calories expended by analyzing the data provided in input. The plan should consist of specific exercises and foods.
    The generated content should be very specific and tailored to the user. The personalized plan has to be made after analyzing the smartwatch data provided as input.
    You have to generate content using the INPUT DATA of the user. The INPUT DATA is a json file with the time period being the primary key and the metrics and activities of the user during the mentioned time period being the values for the key.
    Create a personalized plan where you consider the analysis of user data and output only your suggestions and it should be concise.
    The personalized plan should contain exercise names and the week routine to do it and the diet which the user should follow to stay healthy.

Remember to stay hydrated throughout the day and listen to your body's needs. Adjust the portion sizes according to your activity level and goals.


    INPUT DATA:
    {user_data}

    OUTPUT FORMAT:
    Personalized Fitness Plan:
    - Week Routine:
        - Monday: ....
        - Tuesday: ....
        - Wednesday: ....
        - Thursday: ....
        - Friday: ....
        - Saturday: ....
        - Sunday: ....

    Personalized Diet Plan:
    - Breakfast: ....
    - Snack: ....
    - Lunch: ....
    - Snack: ....
    - Dinner: ....
    - Snack: ....

    '''

    pp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {"role": "system", "content": prompt},
        ],
        max_tokens=2500,
        temperature=0.2
        )

    plan = pp['choices'][0]['message']['content']
    media_directory = settings.MEDIA_ROOT
    file_path = os.path.join(media_directory, 'plan2.txt')
    with open(file_path, 'w') as f:
      f.write(plan)


def get_all_plans(restructured_data):
    output_queue = multiprocessing.Queue()
    p1 = multiprocessing.Process(target=health_recommendation, args=(restructured_data,))
    p1.start()

    time.sleep(2)
    p2 = multiprocessing.Process(target=exercise_recommendation, args=(restructured_data,))

    # Start the processes

    p2.start()

    # Wait for the processes to finish
    p1.join()
    p2.join()

# data = new_json()
# health_recommendation(data)
