from dotenv import load_dotenv
from openai import OpenAI
import os
import requests
import json
#CA:0E:D4:AD:FC:7A:8D:EB
#3A:D7:D4:AD:FC:7B:09:95
#H61A0

load_dotenv()
govee_base_url = "https://developer-api.govee.com/v1"
govee_api_key = os.environ['GOVEE_API_KEY']
devices = ["CA:0E:D4:AD:FC:7A:8D:EB", "3A:D7:D4:AD:FC:7B:09:95"]

devices = [
    {
        "mac":"CA:0E:D4:AD:FC:7A:8D:EB",
        "model": "H61A0",
        "location": "room"
    },
    {
        "mac":"3A:D7:D4:AD:FC:7B:09:95",
        "model": "H61A0",
        "location": "room"
    },
    {
        "mac":"22:C3:38:33:33:67:60:FF",
        "model": "H61C2",
        "location": "desk"
    },
    {
        "mac":"A1:DF:D4:AD:FC:A4:FC:BA",
        "model": "H61A1",
        "location": "living room"
    },
    {
        "mac":"68:69:D4:AD:FC:FF:97:F3",
        "model": "H61A1",
        "location": "living room"
    }

]

def turn_all_lights(state):
    url = govee_base_url + "/devices/control"
    headers = {"Content-Type": "application/json",
               "User-Agent": "PostmanRuntime/7.34.0",
               "Accept": "*/*",
                "Govee-API-Key": govee_api_key}
    
    for device in devices:
        data = {
        "device": device["mac"],
        "model": device["model"],
        "cmd": {
            "name": "turn",
            "value": state
        }}
        x = requests.put(url, headers=headers, json=data) 

      
    return x.text

def change_all_lights_color(color):
    url = govee_base_url + "/devices/control"
    headers = {"Content-Type": "application/json",
               "User-Agent": "PostmanRuntime/7.34.0",
               "Accept": "*/*",
                "Govee-API-Key": govee_api_key}
    
    for device in devices:
        data = {
        "device": device["mac"],
        "model": device["model"],
        "cmd": {
            "name": "color",
            "value": {
                "r": color["r"],
                "g": color["g"],
                "b": color["b"]
            }
        }
        }
        x = requests.put(url, headers=headers, json=data)
    return x.text

def change_room_lights(state, room):
    url = govee_base_url + "/devices/control"
    headers = {"Content-Type": "application/json",
               "User-Agent": "PostmanRuntime/7.34.0",
               "Accept": "*/*",
                "Govee-API-Key": govee_api_key}
    
    for device in devices:
        if device["location"] == room:
            data = {
            "device": device["mac"],
            "model": device["model"],
            "cmd": {
                "name": "turn",
                "value": state
            }
            }
    x = requests.put(url, headers=headers, json=data)
            
    return x.text

def change_light_state(state):
    url = govee_base_url + "/devices/control"
    headers = {"Content-Type": "application/json",
               "User-Agent": "PostmanRuntime/7.34.0",
               "Accept": "*/*",
                "Govee-API-Key": govee_api_key}
    for device in devices:
        data = {
        "device": device,
        "model": "H61A0",
        "cmd": {
            "name": "turn",
            "value": state
        }
        }
        x = requests.put(url, headers=headers, json=data)
    return x.text

def set_color(color):
    url = govee_base_url + "/devices/control"
    headers = {"Content-Type": "application/json",
               "User-Agent": "PostmanRuntime/7.34.0",
               "Accept": "*/*",
                "Govee-API-Key": govee_api_key}
    for device in devices:
        data = {
        "device": device,
        "model": "H61A0",
        "cmd": {
            "name": "color",
            "value": {
                "r": color["r"],
                "g": color["g"],
                "b": color["b"]
            }
        }}
        x = requests.put(url, headers=headers, json=data)
    return x.text
