from decouple import config
import openai
import requests
from file_chooser import choose_a_file
import tkinter as tk


openai.api_key = config("API_KEY")
THE_SCORER_URL = str(config("SCORING_URL"))

message_flow = [
    {
        "role": "system", "content": """
You are an expert at scoring games.
You should calculate the score for the game of cribbage.
"""}
]

hand_desc = choose_a_file("input_crib_hands", ".txt", "Please choose the hand of cards by number:")
print()
message_flow.append({"role": "user", "content": f"Please score my cribbage hand: ```{hand_desc}```\n"})
functions = [
    {
        "name": "get_cribbage_hand_score",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "starter": {
                    "type": "string",
                    "description": "The starter card, e.g. '5H' for 5 of hearts",
                },
                "hand": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "A card in the hand, e.g. and array with cards like '5H' for 5 of hearts"
                    }
                },
                "isCrib": {
                    "type": "boolean", 
                    "description": "Whether the hand is a crib"
                }
            },
            "required": ["starter", "hand", "isCrib"],
        }
    }
]

print(message_flow)
print("\nCalculating the results...\n")

response = openai.ChatCompletion.create(
    model="gpt-4-0613",
    messages=message_flow,
    functions=functions,
    function_call="auto",
    temperature=0,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)
resp_message = response["choices"][0]["message"]

results = ""
if resp_message["function_call"]["name"] == "get_cribbage_hand_score":
    print(resp_message)
    headers = {'Content-type': 'application/json',
               'Accept': 'application/json'}
    results = requests.post(
        url=THE_SCORER_URL, data=resp_message["function_call"]["arguments"],
        headers=headers).json()
else:
    print("Bad things have occured, head for the hills.")
    exit(1)

the_score_details = results["message"]
print(results["message"])

message_flow2 = [
    {
        "role": "system", "content": """
You are an expert at scoring games.
You should take the Cribbage Show score results provided and clarify them in plain English.
Remember to translate the raw results into plain English, for example:
(5, 'H') means 5 of Hearts, 
(11, 'C') means Jack of Clubs etc.
"""}
]
message_flow2.append({"role": "assistant", "content": f"These are the calculated results that need explaining```{the_score_details}```\n"})

response2 = openai.ChatCompletion.create(
    model="gpt-4-0613",
    messages=message_flow2,
    temperature=0,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)
resp_message = response2["choices"][0]["message"]

print()
print(resp_message["content"])

# Create a window
window = tk.Tk()

window.title("Cribbage Score")
response_label = tk.Label(window, text=resp_message["content"], wraplength=500)
response_label.pack()

# Run the window
window.mainloop()

