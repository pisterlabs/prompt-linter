""" Test whether the LLM is able to read values from a graph.
"""
import guidance
import numpy as np

import t2ebm
from t2ebm import graphs
from t2ebm import prompts
import test_util

import json
import os
from datetime import datetime
import copy
import time

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


def graph_largest_jump(graph, x_val):
    """returns the position and the size of the largest jump in the graph"""
    jumps = [
        np.abs(graph.scores[idx] - graph.scores[idx - 1])
        for idx in range(1, len(graph.x_vals))
    ]
    largest_jump_idx = np.argmax(jumps)
    return graph.x_vals[largest_jump_idx][1], jumps[largest_jump_idx]


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(7),
)
def find_largest_jump_in_graph(llm, ebm, feature_name, x_val=None):
    feature_index = ebm.feature_names.index(feature_name)
    graph = graphs.extract_graph(ebm, feature_index)
    graph = graphs.simplify_graph(graph, 0.05)
    graph_as_text = graphs.graph_to_text(graph, max_tokens=3000)
    prompt = prompts.describe_graph(graph_as_text, include_assistant_response=True)
    # if x_val is not provided, choose a random bin and then sample a random value from within that bin
    if x_val is None:
        num_bins = len(graph.x_vals)
        bin_index = np.random.randint(num_bins)
        bin = graph.x_vals[bin_index]
        x_val = np.random.uniform(bin[0], bin[1])
    # the actual value of the graph at x_value
    # text_graph = graphs.text_to_graph(graph_as_text)
    jump_pos, jump_size = graph_largest_jump(graph, x_val)
    # the prompt
    prompt += """\n\n{{#user~}}\nThanks. Within the different intervals, the graph predicts the same mean value. This means that we have discontinuous 'jumps' in the mean of the graph in between the intervals.
    
We are now looking for the biggest jumps in the graph, that is the jumps with the largest absolute magnitude.

We are specifically looking for the position and size of the largest absolute jumps in the graph. We are looking for both positive and negative jumps, in the end we care about the absolute magnitude of the jumps.

What are the 20 largest jumps in the graph? Please consider all parts of the the graph from left to right.\n{{~/user}}\n\n"""
    prompt += """{{#assistant~}}{{gen 'all_jumps' temperature=0.7 max_tokens=3000}}{{~/assistant}}\n\n"""
    prompt += """\n\n{{#user~}}\nThanks. Now, what it the biggest jump in the graph, and what is its position on the x-axis?\n{{~/user}}\n\n"""
    prompt += """{{#assistant~}}{{gen 'larget_jump' temperature=0.7 max_tokens=1000}}{{~/assistant}}\n\n"""
    # print(guidance(prompt, llm=llm))
    # our prompts use guidance, and this is a nice way to print them
    response = guidance(prompt, llm=llm)()
    # print(response["all_jumps"])
    # logging, but only if the folder for the logs exists
    log_folder = "../logs/largest-jump/gpt-3.5-turbo-16k-0613"
    if os.path.exists(log_folder):
        # the current data, including the hour, minute and seconds
        now = datetime.now()
        current_time = now.strftime("%H-%M-%S")
        current_date = now.strftime("%d-%m-%Y")
        logfilename = os.path.join(
            log_folder, f"{current_date}-{current_time}-read-value.txt"
        )
        # log the response as a string
        with open(logfilename, "w") as logfile:
            logfile.write(str(response))
    return jump_pos, jump_size, response["larget_jump"]


if __name__ == "__main__":
    llm = test_util.openai_setup_gpt3_5()
    # llm = test_util.openai_setup_gpt4()
    for dataset_name in test_util.get_avaialble_datasets():
        ebm = test_util.get_ebm(dataset_name)
        _, _, _, _, feature_names = test_util.get_dataset(dataset_name)
        for feature_idx, feature_name in enumerate(feature_names):
            graph = graphs.extract_graph(ebm, feature_idx)
            if graph.feature_type == "continuous":
                print(dataset_name, feature_name)
                print(find_largest_jump_in_graph(llm, ebm, feature_name))
                # sleep for 1 second
                time.sleep(1)
