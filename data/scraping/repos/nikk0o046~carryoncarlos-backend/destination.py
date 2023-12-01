import os
import re
import time
import logging
logger = logging.getLogger(__name__)
import openai

from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.environ.get('OPENAI_API_KEY')


def create_destination_params(user_request : str, user_id : str) -> dict:
    """
    This function takes the user request and the selectedcityID and returns the destination parameters.

    Args:
        user_request (str): The user request.
        selectedCityID (str): The selected city ID.
        user_id (str): The user ID.
    
    Returns:
        dict: The destination parameters.
    """

    start_time = time.time() # start timer to log it later
    logger.debug("[UserID: %s] Creating destination parameters...", user_id)
    
    system_template = """You are an advanced AI agent tasked with identifying as many potential destination airports as possible based on user preferences. Your response should include:

1. An initial thought process or reasoning for the task.
2. An exhaustive list of IATA airport codes matching the criteria, formatted as [XXX,YYY,ZZZ].

For ambiguous destinations, aim for at least 15 to 20 airport codes. Offering more options increases the chances of finding affordable flights for the user. Focus on final destination airports only, excluding connecting airports. Disregard any irrelevant information."""

    human_template = user_request

    # Construct the conversation message list
    message_list = [
        {"role": "system", "content": system_template},
        {"role": "user", "content": human_template}
    ]

    # Request the response from the model
    response = openai.ChatCompletion.create(
      model="ft:gpt-3.5-turbo-0613:personal::8H7hy8ud",
      temperature=0,
      messages=message_list,
    )
    response_content = response.choices[0].message['content']

    logger.debug("[UserID: %s] Destination parameters response: %s", user_id, response_content)

    # Regular expression pattern to match the IATA codes
    pattern = r'\[([A-Za-z,\s]+)\]'

    # Find the matches in the response content
    matches = re.search(pattern, response_content)

    # If a match was found
    if matches:
        # Get the matched string, remove spaces, and split it into a list on the commas
        destination_list = matches.group(1).replace(" ", "").split(',')

        # Create a destination dictionary from the response
        destination_params = {
            'fly_to' : ','.join(destination_list),
        }

    else:
        destination_params = {}

    logger.debug("[UserID: %s] Destination parameters created: %s", user_id, destination_params)
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.debug("[UserID: %s] Function execution time: %s seconds", user_id, elapsed_time)

    return destination_params