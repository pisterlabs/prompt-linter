import openai
import requests


def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""

    # Define the endpoint
    url = "http://localhost:8000/get_weather"

    # Prepare the data
    data = {"location": location, "unit": unit}

    # Send the POST request
    response = requests.post(url, json=data)

    if response.status_code == 200:
        # If the request was successful, return the data
        return response.json()
    else:
        # If the request failed, raise an exception
        raise Exception(f"Request failed with status {response.status_code}.")


# Step 1, send model the user query and what functions it has access to
def run_conversation():
    openai.api_key_path = ".openai-api-key"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {"role": "user", "content": "What's the weather like in Boston?"}
        ],
        functions=[
            {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["location"],
                },
            }
        ],
        function_call="auto",
    )

    message = response["choices"][0]["message"]

    # Step 2, check if the model wants to call a function
    if message.get("function_call"):
        function_name = message["function_call"]["name"]

        # Step 3, call the function
        # Note: the JSON response from the model may not be valid JSON
        function_response = get_current_weather(
            location=message.get("location"),
            unit=message.get("unit"),
        )

        # Step 4, send model the info on the function call and function response
        second_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=[
                {
                    "role": "user",
                    "content": "What is the weather like in boston?",
                },
                message,
                {
                    "role": "function",
                    "name": function_name,
                    "content": function_response,
                },
            ],
        )
        return second_response


print(run_conversation())
