from helper_param import *

import openai
import googlemaps
import io
import json


# Context file path
context_file_path = "../context_handbook.txt"

def read_text_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    return text

prompt = read_text_file(context_file_path)

conversation=[{"role":"system","content":prompt}]

functions = [
        {
            "name": "get_directions",
            "description": "Give direction to some location. ",
            "parameters": {
                "type": "object",
                "properties": {
                    "end_location": {
                        "type": "string",
                        "description": "Location person wants to go to. This is end location. This location can be in University at buffalo",
                    },
                    "start_location": {
                        "type": "string",
                        "description": "This is start location person to start journey from. Return None if start location is not specified ",
                    },
                },
                "required": ["end_location"],
            },
        }, 

        {
            "name": "president_ub",
            "description": "who is president of University of Buffalo",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }, 
        
        {
            "name": "chair_ub",
            "description": "who is Chair of Computer Science department of University of Buffalo",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }, 

        {
            "name": "provost_ub",
            "description": "who is Provost of University of Buffalo",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },

        {
            "name": "Dean_ub",
            "description": "who is Dean of school of engineering and applied science of University of Buffalo",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        
        {
            "name": "VPR_ub",
            "description": "who is VPR, Vice President for Research, of University of Buffalo",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },

        {
            "name": "Coffee",
            "description": "Where Can I find best coffee shop at University at Buffalo",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },

        {
            "name": "Disable",
            "description": "When someone ask to disable authentication i.e to stop audio authentication",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },

        {
            "name": "Enable",
            "description": "When someone ask to enable authentication i.e to start audio authentication",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },

        
        {
            "name": "Wakeup",
            "description": "When someone ask to wakeup",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },

        {
            "name": "Thanks",
            "description": "When someone ask to say Thank you",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },

    ]


def VPR_ub():
    return " Dr. Venu Govindraju is Vice President of Research and Economic Developement at the University at Buffalo. "

def Dean_ub():
    return "Dr. Kemper Lewis is Dean of School of Engineering and Applied Sciences at the University at Buffalo."

def provost_ub():
    return " Dr. Scott Weber is Provost at the University at Buffalo."

def chair_ub():
    return " Dr. Jinhui Xu is a Chair of CSE department at the University at Buffalo."

def president_ub():
    return " Dr. Satish Tripathi is President of the University at Buffalo."

def wakeup():
    return ""

def thanks():
    return ""

def enable():
    return ""

def disable():
    return ""

def Coffee():
    return get_directions("Davis Hall, University at Buffalo", "Student Union, University at Buffalo")

def get_directions(start_location, end_location):
    api_key = ChatGPT_API_KEY
    gmaps = googlemaps.Client(key=api_key)

    # Geocode the start and end locations to get their latitude and longitude
    start_geocode = gmaps.geocode(start_location)
    end_geocode = gmaps.geocode(end_location)

    if not start_geocode or not end_geocode:
        return None

    start_latlng = start_geocode[0]['geometry']['location']
    end_latlng = end_geocode[0]['geometry']['location']

    # Get directions between the start and end locations
    directions = gmaps.directions(start_location, end_location, mode="walking")

    map_image_url = f"https://maps.googleapis.com/maps/api/staticmap?" \
                    f"size=1200x1800&" \
                    f"markers=color:red|label:S|{start_latlng['lat']},{start_latlng['lng']}&" \
                    f"markers=color:green|label:E|{end_latlng['lat']},{end_latlng['lng']}&" \
                    f"path=color:blue|enc:{directions[0]['overview_polyline']['points']}&" \
                    f"key={api_key}"

    return  map_image_url

def gptResponse(question):
    # using the openai api key
    # using the openai api key
    # openai.api_key=openai_key
    data = {
    'question':question,}
    api_url = 'http://128.205.43.183:5110/chat' 
    response = requests.post(api_url, json=data)
    elapsed_time = response.elapsed.total_seconds()
    print(elapsed_time)
    if response.status_code == 200:
        result = response.json()
        answer = result.get('answer', 'No answer provided')
    else:
        print("Error:", response.text)
    # conversation.append({"role":"user","content": question})
    # response=openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo-16k",
    #     messages=conversation,
    #     temperature=0.2,
    #     max_tokens=1000,
    #     top_p=0.2,
    #     functions=functions,
    #     function_call="auto",
    # )
    # response_message = response["choices"][0]["message"]
    # if response_message.get("function_call"):
        
    #     available_functions = { "get_directions": get_directions , "president_ub" : president_ub, "chair_ub": chair_ub, "provost_ub": provost_ub , "Dean_ub": Dean_ub , "VPR_ub" : VPR_ub  }  
    #     function_name = response_message["function_call"]["name"]

    #     if function_name == "get_directions":
    #         fuction_to_call = available_functions[function_name]
    #         function_args = json.loads(response_message["function_call"]["arguments"])

            
    #         if function_args.get("start_location") == None:
    #             s_location = "Davis Hall, University at Buffalo"
            
    #         e_location = function_args.get("end_location")

    #         function_response = fuction_to_call(
    #         start_location=s_location,
    #         end_location= e_location,
    #         )
    #         print(f'Start location is {s_location}')
    #         print(f"---")
    #         print(f'Destination is {e_location}')
        
    #         return "map", function_response
        
    #     elif function_name == "president_ub":
    #         fuction_to_call = available_functions[function_name]  
    #         function_response = fuction_to_call()
    #         return "president", function_response
        
    #     elif function_name == "chair_ub":
    #         fuction_to_call = available_functions[function_name]
    #         function_response = fuction_to_call()        
    #         return "chair", function_response
        
    #     elif function_name == "provost_ub":
    #         fuction_to_call = available_functions[function_name]
    #         function_response = fuction_to_call()        
    #         return "provost", function_response
        
    #     elif function_name == "Dean_ub":
    #         fuction_to_call = available_functions[function_name]
    #         function_response = fuction_to_call()        
    #         return "dean", function_response
        
    #     elif function_name == "VPR_ub":
    #         fuction_to_call = available_functions[function_name]
    #         function_response = fuction_to_call()        
    #         return "vpr", function_response

    # else:

    #     #conversation.append({"role":"assistant","content":response['choices'][0]['message']['content']})
    #     #answer = response['choices'][0]['message']['content']
    return "chat", answer
