"""
THIS SCRIPT IS UNFINISHED

This script will be used to use GPT-4 to create fine-tuning data for the time parameters function.
"""

import os
import openai
import re
import json
import time
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.environ.get('OPENAI_API_KEY')


def create_time_dict(test_case_number : int, user_request : str, current_date : str) -> dict:
    """
    This function takes a user request and the current date and returns parameters to evaluate the time params model.

    Args:
        user_request (str): The user request.
        current_date (str): The current date.

    Returns:
        dict: A dictionary containing test_case_number (int), the GPT-4 response content (str), the extracted time params (dict), the elapsed time (float), the prompt tokens (int), the completion tokens (int) and quality score (float), which is a placeholder.
    """


    start_time = time.time() #start timer to log it later

    #create the prompt templates
    system_template = """API DOCUMENTATION:
departure_date_from, departure_date_to: Range for outbound flight departure (dd/mm/yyyy). 

nights_in_dst_from, nights_in_dst_to: Minimum and maximum stay length at the destination (in nights). Only exclude these if the user is looking for a one-way trip. Otherwise you must make an assumption.

fly_days, ret_fly_days: List of preferred days for outbound and return flights (0=Sunday, 1=Monday, ... 6=Saturday). 

fly_days_type, ret_fly_days_type: Specifies if fly_days/ret_fly_days is for an arrival or a departure flight.

If the user looks for specific dates, set departure_date_from and departure_date_to to a specific date, and match nights_in_dst_from and nights_in_dst_to so that the return day will be correct.

ANSWER INSTRUCTIONS:
Your task is to create parameters specified above based on user information. The parameters will be forwarded to another assistant, who uses them to search flights. Do not come up with any other parameters.
The output should include both:
1) Thought: Thinking out loud about the user's needs and the task.
2) Markdown code snippet formatted in the following schema, including the leading and trailing "\`\`\`json" and "\`\`\`":

```json
{
    "key1": value1  // Define relevant values. Only use keys mentioned in the API documentation. 
    "key2": value2
}
```"""

    #example 1
    userExample1 = "Current date: 10/07/2023\nInfo: Origin: London, GB | Destination: Paris, FR | Departure: Next month's Friday| Duration: Weekend | Flights: Any"

    botExample1 = """Thought: User wants to leave on a Friday next month (August) and stay for two nights.
```json
{
    "departure_date_from": "01/08/2023",
    "departure_date_to": "31/08/2023",
    "fly_days": 5,
    "fly_days_type": "departure",
    "nights_in_dst_from": 2,
    "nights_in_dst_to": 2,
    "ret_fly_days": 0,
    "ret_fly_days_type": "departure"
}
```"""

    #example 2
    userExample2 = """Current date: 01/01/2024\nInfo: Origin: San Francisco, US | Destination: Anywhere abroad | Departure: March | Duration: About a week | Flights: Any"""

    botExample2 = """Thought: setting departure dates for next March, lasting about a week which translated to 6-8 nights.
```json
{
"departure_date_from": "01/03/2024",
"departure_date_to": "31/03/2024",
"nights_in_dst_from": 6,
"nights_in_dst_to": 8
}
```"""

    #example 3
    userExample3 = "Current date: 10/08/2023\nInfo: Origin: Los Angeles, US | Destination: Miami, US | Departure: October's Long Weekend | Duration: 3 days | Flights: direct"

    botExample3 = """Thought: Long weekend usually means three days. Possible departure days are Thursday and Friday. Possible return flight days are Sunday or Monday.
```json
{
    "departure_date_from": "01/10/2023",
    "departure_date_to": "31/10/2023",
    "nights_in_dst_from": 3,
    "nights_in_dst_to": 3,
    "fly_days": [4, 5],
    "ret_fly_days": [0, 1]
}
```"""

    #example 4
    userExample4 = "Current date: 10/04/2023\nInfo: Origin: Chicago, US | Destination: Paris, FR | Departure: Summer | Flights: One-way"

    botExample4 = """Thought: The user only needs an outbound flight to Paris, which should be anytime in the summer months (June, July, August). Because it is a one-way trip, nights_in_dst-parameters must be excluded. 
```json
{
    "departure_date_from": "01/06/2023",
    "departure_date_to": "31/08/2023"
}
```"""

    #example 5
    userExample5 = "Current date: 10/07/2023\nInfo: Origin: Boston, US | Destination: Abroad | Activity: not specified | Flights: 4th of October to 8th of October"

    botExample5 = """Thought: The user wants the outbound flight on 4th of October, so we set the departure window (departure_date_from and departure_date_to) to a single day. The return is on 8th of October, so the stay is exactly 4 nights. Therefore we set both nights_in_dst_from and nights_in_dst_to to 4.
```json
{
    "departure_date_from": "04/10/2023",
    "departure_date_to": "04/10/2023",
    "nights_in_dst_from": 4,
    "nights_in_dst_to": 4
}
```"""

    human_template = f"Current date: {current_date}\nInfo: {user_request}"

    # Construct the conversation message list
    message_list = [
        {"role": "system", "content": system_template},
        {"role": "user", "content": userExample1},
        {"role": "assistant", "content": botExample1},
        {"role": "user", "content": userExample2},
        {"role": "assistant", "content": botExample2},
        {"role": "user", "content": userExample3},
        {"role": "assistant", "content": botExample3},
        {"role": "user", "content": userExample4},
        {"role": "assistant", "content": botExample4},
        {"role": "user", "content": userExample5},
        {"role": "assistant", "content": botExample5},
        {"role": "user", "content": human_template}
    ]

    # Request the response from the model
    response = openai.ChatCompletion.create(
      model="gpt-4",
      temperature=0,
      messages=message_list,
    )
    response_content = response.choices[0].message['content']

    # Extract the json string using regular expressions
    json_str = re.search(r"\{.*\}", response_content, re.DOTALL).group()

    # Convert the json string to a Python dictionary

    time_params = json.loads(json_str)

    end_time = time.time()
    elapsed_time = end_time - start_time

    output_dict = {
        "test_case_number": test_case_number,
        "response_content": response_content,
        "time_params": time_params,
        "elapsed_time": elapsed_time,
        "prompt_tokens": response.usage["prompt_tokens"],
        "completion_tokens": response.usage["completion_tokens"],
        "quality": 1.0 # Default is 1.0, which means as good as it gets. Bug is 0.0. If it doesn't crash but isn't perfect, use 0.5.
    }

    return output_dict


if __name__ == "__main__":
    def save_to_file(data, filename="../data/time_answers_raw.json"):
        with open(filename, "a") as file:
            json.dump(data, file)
            file.write("\n")


    # Read the JSON file
    with open("../data/test_cases.json", "r") as file:
        test_cases = json.load(file)

    # Loop through the test cases
    for test_case in test_cases:
        user_request = test_case["user_request"]
        date = test_case["date"]
        test_case_number = test_case["test_case_number"]
        print(f"Test number: {test_case_number}")
        result = create_time_dict(test_case_number, user_request, date)
        save_to_file(result)

