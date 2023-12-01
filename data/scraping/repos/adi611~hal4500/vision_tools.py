import cv2 
import json
import os
import dotenv
import openai
import asyncio

dotenv.load_dotenv()

openai.api_key = os.getenv("OPEN_AI_API")


path = 'output.json'
def what_am_i_holding():
    visual_input = None
    #image = cv2.imread('current.jpg')
    with open(path , 'r') as f:
        visual_input = json.load(f)
    
    if visual_input :
        if visual_input['left_hand_object'] :
            xmin = visual_input['left_hand_object']['xmin']
            ymin = visual_input['left_hand_object']['ymin']
            xmax = visual_input['left_hand_object']['xmax']
            ymax = visual_input['left_hand_object']['ymax']
            #roi = image[int(ymin):int(ymax), int(xmin):int(xmax)]
            #cv2.imwrite('left_hand_object.jpg', roi)

        if visual_input['right_hand_object'] :
            xmin = visual_input['right_hand_object']['xmin']
            ymin = visual_input['right_hand_object']['ymin']
            xmax = visual_input['right_hand_object']['xmax']
            ymax = visual_input['right_hand_object']['ymax']
            #roi = image[int(ymin):int(ymax), int(xmin):int(xmax)]
            #cv2.imwrite('right_hand_object.jpg', roi)


        

        chat_response = None
        if visual_input['left_hand_object'] and visual_input['right_hand_object'] :
            chat_response = f'You are holding a {visual_input["left_hand_object"]["name"]} in your left hand and a {visual_input["right_hand_object"]["name"]} in your right hand'
        elif visual_input['left_hand_object'] :
            chat_response = f'You are holding a {visual_input["left_hand_object"]["name"]} in your left hand'
        elif visual_input['right_hand_object'] :
            chat_response = f'You are holding a {visual_input["right_hand_object"]["name"]} in your right hand'
        else :
            chat_response = 'You are not holding anything'
        
        return chat_response


