from openai import OpenAI
from dotenv import load_dotenv
import os
from PIL import Image
import base64
import pandas as pd
import json
import pytesseract
import cv2
import numpy as np
def image_to_base64(image_path):
    try:
        # Open the image using Pillow
        with open(image_path, "rb") as img_file:
            # Read the image binary data
            image_binary = img_file.read()
            
            # Convert the binary data to base64
            base64_data = base64.b64encode(image_binary).decode("utf-8")
            
            # Convert bytes to a UTF-8 encoded string
            
            return base64_data
    except Exception as e:
        print(f"Error: {e}")
        return None
    
def performOCR():

    # Load the image from file
    img_path = './download.jpg'
    img = cv2.imread(img_path)

    # Convert the image to black and white
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)

    img = cv2.dilate(img, kernel, iterations=1)

    img = cv2.erode(img, kernel, iterations=1)
    # Use tesseract to do OCR on the image
    text = pytesseract.image_to_string(img)

    return text

def get_bill_details(client):
    invoice_text = performOCR()
    prompt = f"Given the image of the invoice and the text from the PyTesseractOCR Model below, *MATCH* and convert the text into JSON format.\nInvoice Text: {invoice_text}."
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image",
                        "image": image_to_base64("download.jpg"),
                    },
                ],
            }
        ],
        max_tokens=1000,
    )

    return (response.choices[0].message.content)

def add_item_to_dataframe(item, quantity, price):
    global df
    new_row = pd.DataFrame([[item, quantity, price, quantity * price]], columns=['Item', 'Quantity', 'Price', 'Total'])
    df = pd.concat([df, new_row], axis=0, ignore_index=True)
    return json.dumps(df.to_json(orient='records')) # Return the DataFrame as a JSON string

df = pd.DataFrame(columns=['Item', 'Quantity', 'Price', 'Total'])
def get_dataframes_using_convo(client, bill_details):
    
    messages = [{"role": "system", "content": "Given a JSON of items, create multiple add_item_to_dataframe function calls for each items given by the user."}]
    messages.append({"role": "user", "content": bill_details})
    tools = [
        {
            "type": "function",
            "function": {
                "name": "add_item_to_dataframe",
                "description": "Add given items and its details into a pandas dataframe",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "item": {
                            "type": "string",
                            "description": "The name of the item",
                        },
                        "quantity": {
                            "type": "integer",
                            "description": "The number of the items bought not the size or weight",
                        },
                        "price": {
                            "type": "integer",
                            "description": "The price of the one unit of the item, not the discount or total price(rate per unit)"
                        },
                    },
                    "required": ["item", "quantity", "price"],
                },
            }
        }
    ]

    response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=messages,
            tools=tools,
            tool_choice="auto",  # auto is default, but we'll be explicit
        )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    print(tool_calls)
    if tool_calls:
            # Step 3: call the function
            # Note: the JSON response may not always be valid; be sure to handle errors
            available_functions = {
                "add_item_to_dataframe": add_item_to_dataframe,
            }  # only one function in this example, but you can have multiple
            messages.append(response_message)  # extend conversation with assistant's reply
            # Step 4: send the info for each function call and function response to the model
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                function_response = function_to_call(
                    item=function_args.get("item"),
                    quantity=function_args.get("quantity"),
                    price=function_args.get("price")
                )
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    }
                )  # extend conversation with function response
            return (df)
            """
            second_response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=messages,
            )  # get a new response from the model where it can see the function response"""
"""
load_dotenv()
client = OpenAI()
bill_details = get_bill_details(client)
df = get_dataframes_using_convo(client, bill_details)
print(df)
df.to_csv('bill.csv', index=False)"""