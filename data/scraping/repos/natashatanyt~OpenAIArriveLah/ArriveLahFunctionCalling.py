import requests
import json
import openai
import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv('API_KEY')

# List all functions and output that GPT can call
function_list = [
    {
        "name": "getBusServiceInfo",
        "description": "Gets information about a specific bus number at a specific stop",
        "parameters": {
            "type": "object",
            "properties": {
                "busNumber": {
                    "type":"string",
                    "description": "The bus number, such as 963, 7, or 81",
                },
                "busStopCode": {
                    "type": "string",
                    "description": "The 5-digit bus stop code, such as 03129 or 46321",
                },
                "nextTiming": {
                    "type": "string",
                    "description": "The time of the immediate next bus arrival at that stop, such as 2023-06-16T11:01:59+08:00",
                },
                "load": {
                    "type": "string",
                    "description": "How crowded the bus is",
                    "enum": ["SEA", "SDA", "LSD"],
                },
                "vehicleType": {
                    "type": "string",
                    "description": "What type of bus",
                    "enum": ["SD", "DD", "BD"],
                },
            },
            "required":["busNumber", "busStopCode", "nextTiming", "load", "vehicleType"],
        },
    }, 
]

# To get the bus number from the API
def search(Array, ServiceNo):
    for i in Array:
        if i["no"] == ServiceNo:
            return i


# To obtain all data about a specific bus at a specific bus stop
def getBusServiceInfo(BusStopCode, ServiceNo):
    """Get information about the busses serving a bus stop"""
    url = "https://arrivelah2.busrouter.sg/?id="+BusStopCode

    payload = {}
    headers = {}

    response = requests.request("GET", url, headers=headers, data=payload)
    output = response.json()

    tempArray = output["services"]

    busInfo = search(tempArray, ServiceNo)

    returnInfo = {
        "busNumber": ServiceNo,
        "busStopCode": BusStopCode,
        "nextTiming": busInfo["next"]["time"],
        "load": busInfo["next"]["load"],
        "vehicleType": busInfo["next"]["type"],
    }

    return json.dumps(returnInfo)

# Construct and obtain the first response from GPT
def GPTFirstResponse(userQuery): 
    return openai.ChatCompletion.create(
        model = "gpt-3.5-turbo-0613",
            # messages is a history of all the messages - so you can chain responses together
            messages=[{"role": "user", "content": (userQuery)}],
            functions=function_list,

            function_call="auto",
    )

# Construct and obtain the second response from GPT (natural language)
def GPTSecondResponse(constructedQuery, ai_response, function_response, calledFunction):
    return openai.ChatCompletion.create(
        model = "gpt-3.5-turbo-0613",

        # here, we want to chain the responses
        # we want what the user inputs, what the AI responds with
        # and also what will be the final GPT response
        messages=[
            {"role": "user", "content": (constructedQuery)},
            ai_response,
            {
                "role": "function",
                "name": calledFunction,
                "content": function_response,
            },
        ],
    )

# UI 
st.title("ArriveLah Function Calling")

st.subheader("Get information about your transport queries in Singapore.")

st.write("Currently, this supports queries about bus arrival timings at bus stops, such as \"When does bus 175 reach bus stop 10331?\"")

userQuery = st.text_input("Enter your query: ")

if st.button("Submit query!"):

    first_response = GPTFirstResponse(userQuery)

    ai_response = first_response["choices"][0]["message"]
    print(first_response)
    print(ai_response)
    jsonOutput = json.loads(ai_response["function_call"]["arguments"])
    calledFunction = ai_response["function_call"]["name"]

    # Checks if the first call uses a specific function
    match (calledFunction):
        case "getBusServiceInfo":
            BusStopCode = jsonOutput["busStopCode"]
            ServiceNo = jsonOutput["busNumber"]
            function_response = getBusServiceInfo(BusStopCode, ServiceNo)

            constructedQuery = "Bus Stop Number: " + BusStopCode + " and Bus Service Number: " + ServiceNo
       

    
    natural_response = GPTSecondResponse(constructedQuery, ai_response, function_response, calledFunction)
    st.write(natural_response["choices"][0]["message"]["content"])
