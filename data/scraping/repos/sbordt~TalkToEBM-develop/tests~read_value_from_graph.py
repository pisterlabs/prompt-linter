""" Test whether the LLM is able to read values from a graph.
"""
import guidance
import numpy as np

import t2ebm
from t2ebm import graphs
from t2ebm import prompts
import test_util

import os
from datetime import datetime
import copy
import time

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


def graph_mean(graph, x_val):
    """Returns the mean of the graph at x_val."""
    # find the bin that x_val is in
    for idx, x_bin in enumerate(graph.x_vals):
        if x_val >= x_bin[0] and x_val <= x_bin[1]:
            bin_index = idx
            break
    # return the mean of that bin
    return graph.scores[bin_index]


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(7),
)
def read_value_from_graph(llm, ebm, feature_name, x_val=None):
    feature_index = ebm.feature_names.index(feature_name)
    graph = graphs.extract_graph(ebm, feature_index)
    graph = graphs.simplify_graph(graph, 0.05)
    prompt = prompts.describe_graph(
        graphs.graph_to_text(graph, max_tokens=3000), include_assistant_response=True
    )
    # if x_val is not provided, choose a random bin and then sample a random value from within that bin
    if x_val is None:
        num_bins = len(graph.x_vals)
        bin_index = np.random.randint(num_bins)
        bin = graph.x_vals[bin_index]
        x_val = np.random.uniform(bin[0], bin[1])
    # the actual value of the graph at x_value
    y_val = graph_mean(graph, x_val)
    # the prompt
    prompt += (
        "\n\n{{#user~}}\nThanks. What is the mean value of the graph at "
        + f"{x_val:.4f}"
        + "?\n{{~/user}}\n\n"
    )
    prompt += """{{#assistant~}}{{gen 'value' temperature=0.0 max_tokens=100}}{{~/assistant}}\n\n"""
    # run the prompts against the llm
    response = guidance(prompt, llm=llm)()
    # logging, but only if the folder for the logs exists
    log_folder = "../logs/read-value/gpt-3.5-turbo-16k-0613"
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
    return x_val, y_val, response["value"]


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
                # read 3 values at random positions from the graph
                for _ in range(3):
                    print(read_value_from_graph(llm, ebm, feature_name))
                    # sleep for 1 second to avoid rate limiting
                    time.sleep(1)
