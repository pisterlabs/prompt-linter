"""
This module is handles the timeline generation for the narrative. It defines several functions to facilitate
the creation, parsing, and correction of timestamp data. The timestamps are crucial in tracking the actions of different
characters at specific moments in the story.

Key Functions:
- correct_json_thread: Corrects JSON string representation of a timestamp in a separate thread.
- fix_json: Initiates threads to correct JSON strings representing timestamps.
- parse_timeline: Parses JSON strings into Timestamp objects and updates the plot's timeline.
- write_timeline_template: Constructs a template for generating a single timestamp in the narrative.
- write_timeline: Handles the generation of the entire timeline, iterating through timestamps, and ensuring consistency
in the narrative flow.
"""

import openai
import os
from story_elements.timestamp import Timestamp
from LLM_functions.write_timestamp import write_timestamp_func
import json
import threading

openai.api_key = os.getenv("OPENAI_API_KEY")


def correct_json_thread(i, timestamp, results):
    """
    Calls GPT to correct the JSON string in a separate thread.

    Args:
        i (int): The index of the timestamp.
        timestamp (str): The JSON string representing the timestamp.
        results (list): The list to store the corrected JSON string.
    """
    prompt = ("You fix JSON strings if they are not correctly formatted. If the string you receive is already "
              "formatted, you just return the string again. You only return JSON, no words.\n\n")
    prompt += f"{timestamp}\n\n"

    corrected_timestamp = openai.ChatCompletion.create(
        model='gpt-3.5-turbo-16k-0613',
        messages=[{'role': 'system', 'content': prompt}],
    )

    corrected_timestamp = corrected_timestamp.choices[0].message['content']
    results[i] = corrected_timestamp


def fix_json(timestamps):
    """
    Uses RCI (another call to GPT) to correct the LLM-outputted JSON strings.

    Args:
        timestamps (list): A list of JSON strings representing timestamps.

    Returns:
        list: A list of corrected JSON strings.
    """
    threads = []
    results = [None] * len(timestamps)

    for i, timestamp in enumerate(timestamps):
        thread = threading.Thread(target=correct_json_thread, args=(i, timestamp, results))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return results


def parse_timeline(timestamps, plot):
    """
    Parses the LLM-outputted JSON strings into Timestamp objects and adds them to the Plot object's timeline.

    Args:
        timestamps (list): A list of JSON strings representing timestamps.
        plot (Plot): The Plot object to which the timestamps will be added.
    """
    for timestamp in timestamps:
        timestamp_dict = json.loads(timestamp)
        time = timestamp_dict["time"]
        suspect_1_action = timestamp_dict["suspect_1_action"]
        suspect_2_action = timestamp_dict["suspect_2_action"]
        suspect_3_action = timestamp_dict["suspect_3_action"]
        suspect_4_action = timestamp_dict["suspect_4_action"]
        timestamp_obj = Timestamp(time, [suspect_1_action, suspect_2_action, suspect_3_action, suspect_4_action])
        plot.add_timestamp_to_timeline(timestamp_obj)


def write_timeline_template(timestamp_number, background_info, shared_events, previous_timestamp, response_format):
    """
    Constructs the template for writing a single timestamp.

    Args:
        timestamp_number (int): The number indicating the current timestamp.
        background_info (str): The background information about the story.
        shared_events (SharedEvent): The shared events object containing interaction details.
        previous_timestamp (str): The JSON string representing the previous timestamp.
        response_format (str): The desired format for the response.

    Returns:
        str: The constructed template.
    """
    timeline_base = ("You generate full timelines in 15 minute intervals. These timeline will detail the actions and "
                     "experiences of each of the four suspects during the event.\n\n")

    timeline_base += f"You are currently writing timestamp {timestamp_number} out of 10.\n\n"

    timeline_base += "BACKGROUND INFO:\n\n"
    timeline_base += f"{background_info}\n\n"

    # Adds the shared events to the template if they exist
    if shared_events.interaction_id is not None:
        shared_interaction = shared_events.get_interaction_pair()

        timeline_base += (f"THE FOLLOWING SUSPECTS MUST INTERACT AND ACKNOWLEDGE EACH-OTHER AT THIS TIMESTAMP "
                          f"({timestamp_number}/10). Their actions must contain each other's names:\n")
        timeline_base += f"{shared_interaction}\n\n"

    # Adds the previous timestamp to the template if it exists
    if previous_timestamp is not None:
        timeline_base += f"PREVIOUS TIMESTAMP (Yours must be 15 minutes after this):\n"
        timeline_base += f"{previous_timestamp}\n\n"

    timeline_base += ("Fill in all the details, weaving a convincing, continuous narrative, otherwise you die.\n ALL "
                      "SUSPECTS (suspect 1, suspect 2, suspect 3, AND suspect 4) MUST HAVE AN ACTION AT EACH "
                      "TIMESTAMP.\n Detail how the murderer plots and carries out their crime; do not write about the "
                      "investigation. \n (ALL TIMES MUST BE IN FORMAT HH:MM AM/PM) \n\n")

    timeline_base += (f"TIMESTAMP {timestamp_number} OUT OF 10 (YOU MUST RETURN YOUR ANSWER USING "
                      f"write_single_timestamp()):")
    return timeline_base


def write_timeline(plot, response_format):
    """
    Handles timeline generation, generates timestamps in 15-minute intervals.

    Args:
        plot (Plot): The Plot object containing the story details.
        response_format (str): The desired format for the response.
    """
    timestamps = []

    with open("../timestamps.txt", "a") as file:
        template = write_timeline_template(1, plot.get_summary(), plot.shared_interactions[0], None, response_format)
        print(template)

        last_timestamp = openai.ChatCompletion.create(
            model='gpt-3.5-turbo-16k-0613',
            messages=[{'role': 'system', 'content': template}],
            functions=write_timestamp_func,
            function_call={'name': 'write_single_timestamp'},
        )

        last_timestamp = last_timestamp.choices[0].message.function_call.arguments
        timestamps.append(last_timestamp)

        file.write(f"Prompt:\n{template}\nGenerated Timestamp:\n{last_timestamp}\n\n")

        for i in range(3, 11):
            template = write_timeline_template(i, plot.get_summary(), plot.shared_interactions[i - 1], last_timestamp,
                                               response_format)
            print(template)

            last_timestamp = openai.ChatCompletion.create(
                model='gpt-3.5-turbo-16k-0613',
                messages=[{'role': 'system', 'content': template}],
                functions=write_timestamp_func,
                function_call={'name': 'write_single_timestamp'},
            )

            last_timestamp = last_timestamp.choices[0].message.function_call.arguments
            timestamps.append(last_timestamp)

            file.write(f"Prompt:\n{template}\nGenerated Timestamp:\n{last_timestamp}\n\n")
            print(last_timestamp)

    timestamps = fix_json(timestamps)

    for timestamp in timestamps:
        print(timestamp)
    parse_timeline(timestamps, plot)
