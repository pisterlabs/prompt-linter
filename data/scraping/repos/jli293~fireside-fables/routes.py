from flask import Blueprint, render_template, request
from .models import Result
from dotenv import load_dotenv
import os
import time
from openai import OpenAI

# Load the environment variables
load_dotenv()

# Create a new blueprint
routes = Blueprint('routes', __name__)

# Set the OpenAI API key
OpenAI.api_key = os.getenv('OPENAI_API_KEY')

# Create a new OpenAI client
client = OpenAI()


def generate_bonfire_story(elements):
    """
    Takes in a list of elements and returns a story from GPT-3.5 Turbo Instruct
    :param elements: list of elements provided by the user
    :return: story
    """

    # Story prompt for GPT-3.5 Turbo Instruct
    story_prompt = "Create a 150-word bonfire story with the following elements: "
    story_prompt = story_prompt + elements
    story_prompt += ".\n\n"

    # Call the OpenAI API to generate a story
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=story_prompt,
        temperature=0.7,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    # Add a delay to prevent exceeding the API rate limit
    time.sleep(1)

    return response.choices[0].text.strip()


# Create a list to store the history data
history_data = []


@routes.route('/', methods=['GET', 'POST'])
def home():
    """
    Home page
    :return: response_view template
    """
    # for GET request, return response_view template
    if request.method == 'GET':
        query = request.args.get('query')
        if query == "" or query is None:
            return render_template('response_view.html')

        # Generate bonfire story
        response = generate_bonfire_story(query)

        data_list = []

        # Create a new Result object
        query_message = Result(message_type="other-message float-right",
                               message=query)

        # Create a new Result object
        response_message = Result(message_type="my-message",
                                  message=response)

        # Add the Result objects to the list
        data_list.append(query_message)
        data_list.append(response_message)

        # Add the Result objects to the history list
        if len(history_data) < 4:
            history_data.append(query_message)
            history_data.append(response_message)

        return render_template('response_view.html', results=data_list)

    # for other requests, return history template
    else:
        # # Create a new list to store history data to render
        # history_list = []
        #
        # # If there are more than 4 elements in the history_data list,
        # if len(history_data) > 6:
        #     # add the first four elements to the history_list
        #     history_list = history_data[:4]
        #
        #     # add the last two elements
        #     history_list.append(history_data[-2])
        #     history_list.append(history_data[-1])
        #
        # # Otherwise, add all elements to the history_list
        # else:
        #     history_list = history_data
        #
        # return render_template('history.html', results=history_list)

        # Return the history template
        return render_template('history.html', results=history_data)
