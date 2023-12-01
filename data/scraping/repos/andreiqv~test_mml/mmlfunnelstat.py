"""

GPT result: {'status': 'success', 'error_source': 'chatgpt', 'answer': '-06-01/2023-06-02": 2,\n                "2023-06-02/2023-06-03": 1,\n                "2023-06-03/2023-06-04": 2,\n                "2023-06-04/2023-06-05": 2,\n                "202', 'input_text': 4294}

- Question: Which week has the lowest conversion rate? And what is the value? Return a range of values for 7 days
- Answer: -06-01/2023-06-02": 2,
                "2023-06-02/2023-06-03": 1,
                "2023-06-03/2023-06-04": 2,
                "2023-06-04/2023-06-05": 2,
                "202


"""

import os
import sys
import re
import json
import openai
import time
import logging
import multiprocessing
# import Levenshtein

# testgold
#openai.organization = "org-BbIjJDdeiFQjD7KKmsfCMRrI"
#openai.api_key = ""

from gpt4allj import Model

MODEL_PATH = "/home/ubuntu/tmp/gpt4all/ggml-gpt4all-j-v1.3-groovy.bin"
print(f"Loading model from {MODEL_PATH}...")
GPT_MODEL = Model(MODEL_PATH)
print(f"The model has been loaded.")


prefix0 = """
You can some data and statistics below.
Try to write a brief answer on the following question about the data.
Here is the question from a user about the data:
"""

# What anomaly can you find in this data?

prefix1 = """
---
Write a short answer to him and some short summary on the data below.
Try to keep the answer short, maximum 2-3 sentences if the user don't ask you to write more.

Here is a description of values

The following items are calculated per date bin for a funnel:

data_per_dtbin: contains event data per date bin in the period requested
For each event for each step of the funnel, the following event items are present:
- browser
- city
- continent
- country
- device_type
- distinct_id: PostHog distinct_id
- dt: the timestamp
- ip_address
- lat
- lon
- location: city and country
- os: operating system
- session_id: PostHog session ID
- time_of_day: a 3 hour bin for the time of day when this event was noted
time_zone
- xpath: the funnel element that was interacted with
- summary_per_dtbin: contains summary data per date bin for a funnel
for each date bin, the following items are present:
- finishing_browsers: all unique browsers that finished the funnel and their counts
- finishing_device_types: all unique device types that finished the funnel and their counts
- f-inishing_ip_addresses: all unique IP addresses that finished the funnel and their counts
- finishing_locations: all locations that finished the funnel and their counts
- finishing_operating_systems: all operating systems that finished the funnel and their counts
- finishing_people: all unique session_ids that finished the funnel,
- finishing_people_count: the count of the finishing people,
- finishing_times_of_day: all unique time of day bins that finished the funnel and their counts,
- starting_*: the same items as finishing_* above, but for the people starting the funnels,
- funnel_outcomes: details for each session ID that started the funnel:
- finished: True if this session ID finished the funnel
- finishing_*: the finishing items for this session ID as finishing_* above
- starting_*: the starting items for this session ID as starting_* above

The following items are available as aggregate statistics for all funnels:
For each funnel by name we have:
- conversion_rate: overall conversion rate for the requested date period (n_finished / n_started)
- conversion_rate_by_time: for each date bin in the period
- n_finishes: the overall number of unique people finishing this funnel in the requested date period
- n_finishes_by_time: for each date bin in the period
- n_starts: the overall number of unique people starting this funnel in the requested date period,
- n_starts_by_time: for each date bin in the period

overall funnel rankings as funnel_rankings item:
- conversion_rate: ranked list of funnels by conversion rate
- conversion_rate_diff: ranked list of funnels by the difference in conversion rate at start of date period and conversion rate at end of date period
- n_starts: ranked list of funnels by number of people starting the funnel
- n_starts_diff: ranked list of funnels by the difference in number of starting people at start of date period and number of starting people at end of date period

"""

postfix = """
The end of the document. 
Return only data in JSON format, don't return additional comments.
"""


def process_text_with_gpt(request):

    answer = GPT_MODEL.generate(request,
        n_predict=300
    )

    return answer


def parse_gpt_answer(gpt_answer):

    try:
        data = json.loads(gpt_answer)
        #print("name:", data.get("name"))
        #print("side:", data.get("side"))
        #print("companies:", data.get("companies"))
        result = {
            "status": "success",
            "error_source": "",
            "data": data,
        }
        return result

    except ValueError:
        start_pos = gpt_answer.find('{')
        end_pos = gpt_answer.rfind('}')
        
        if start_pos != -1 and end_pos > start_pos:
            json_str = gpt_answer[start_pos:end_pos+1]
            try:
                data = json.loads(json_str)
                result = {
                    "status": "success",
                    "error_source": "",
                    "data": data,
                }
                return result

            except ValueError:
                logging.warning("Can not parse the GPT answer.")
                #print("json_str:", json_str)

        else:
            result = {
                "status": "failure",
                "error_source": "json_parsing",
                "data": None,
            }
            return result

    return {
                "status": "failure",
                "error_source": "unknown_error",
                "data": None,
            }


def remove_empty_lines(text):
    lines = text.split('\n')
    new_lines = []

    empty_line_count = 0
    for line in lines:
        if line.strip() == '':
            empty_line_count += 1
            if empty_line_count > 1:
                empty_line_count -= 2
                new_lines.append(line)
        else:
            empty_line_count = 0
            new_lines.append(line)

    return '\n'.join(new_lines)


def process_single_request(question, data):

    data = data[:1000]
    # input_text = remove_empty_lines(input_text)

    request = prefix0 + question + prefix1 + data # + postfix

    request = "Here is the input data:\n" + data + prefix0 + question

    try:
        answer = process_text_with_gpt(request)
        #print("Full response:", answer)
        result = {
            "status": "success",
            "error_source": "chatgpt",
            "answer": answer
        }

    except Exception as ex:
        answer = None
        result = {
            "status": "failure",
            "error_source": "chatgpt",
            "answer": None
        }
        print(request)
        raise Exception(ex)

    #if answer:
    #    result = parse_gpt_answer(answer)

    #print("-----------------------")
    #print(request)
    #with open("_input_text.txt", "wt") as fp:
    #    fp.write(request)
    #print("-----------------------")
    #result["input_text"] = len(request)
    #print("GPT result:", result)

    return result


if __name__ == "__main__":

    with open("project_funnel_aggstats.json") as fp:
        data = fp.read()

    questions = [
        "What anomaly can you find in this data?",
        "What is the conversion rate in percent?",
        "When the flows were maximum and minimum?",
        #"When the flows were maximum and minimum? Write a list of dates in the format like '10 Jan 2000'",
        #"What can you say about retention rate by time?",
        #"How many sessions were started at 2023-06-12/2023-06-13?",
        #"Do you think this data is correct?",
        #"What else can you say about this data?",
    ]

    questions22 = [
        #"Discribe changes/trend in conversion rate by time",
        #"What is max value for n_starts_by_time?"
        #"What is avarage value for n_starts_by_time?"
        "Which week has the lowest conversion rate? And what is the value? Return a range of values for 7 days"
    ]

    for question in questions:
        t1 = time.time()
        res = process_single_request(question, data)
        t2 = time.time()
        print(f"\n- Question: {question}")
        print("- Answer:", res.get("answer"))
        print("Time: {:.2f} sec.".format(t2 - t1))

        
        #with open("_output.json", "wt") as fp:
        #    json.dump(res, fp, indent=4)