"""
Get the order of events from a user transcript.
"""

import openai
import os


def get_order_of_events(transcript):
    """ Get the order of events from a user transcript. """
    conversation = [
        {"role": "system", "content": "You are creating an order of events as they happened, based only on what the user has told you. Highlight if any injuries were mentioned."},
        {"role": "assistant", "content": "Can you provide a detailed account of the car accident, including the order of events and any injuries?"},
        {"role": "user", "content": transcript}
    ]

    # Get API key.
    api_key = os.getenv("OPENAI_API_KEY")

    print("Getting order of events.")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # You can choose the most suitable engine
        # Define a list of messages
        messages=conversation,
        max_tokens=200,  # Set a limit on the response length
        api_key=api_key
    )

    return response['choices'][0]['message']['content']


if __name__ == "__main__":
    transcript = """
        I was driving on the main road an hour ago. No one was injured. I was driving the car. 
        I was driving along the road and they pulled out from the right side of me. It was by a set of traffic lights.
        I was driving at 30mph. I was really worried because I missed my appointment.
        The car is damaged on the right side. The wheel is making a strange noise. The car is not drivable.
    """

    order_of_events = get_order_of_events(transcript=transcript)
    print(order_of_events)
