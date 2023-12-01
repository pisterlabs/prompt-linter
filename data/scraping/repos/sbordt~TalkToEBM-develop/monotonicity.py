""" Test whether the LLM is able to decide monotonicity of a graph.
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


def is_monotone_increasing(graph):
    """Return: True/False"""
    # find the bin that x_val is in
    for idx, x_bin in enumerate(graph.x_vals):
        if idx > 0 and idx < len(graph.x_vals):
            if graph.scores[idx] < graph.scores[idx - 1]:
                return False
    return True


def is_monotone_decreasing(graph):
    """Return: True/False"""
    # find the bin that x_val is in
    for idx, x_bin in enumerate(graph.x_vals):
        if idx > 0 and idx < len(graph.x_vals):
            if graph.scores[idx] > graph.scores[idx - 1]:
                return False
    return True


def invert_graph(graph: t2ebm.graphs.EBMGraph):
    """Returns a new graph with the y-axis inverted."""
    new_graph = copy.deepcopy(graph)
    new_graph.scores = -new_graph.scores
    return new_graph


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(7),
)
def decide_monotonicity(llm, graph_as_text: str):
    # prompt setup
    prompt = prompts.describe_graph(
        graph_as_text,
        include_assistant_response=True,  # , special_task_description="Pay special attention to the monotonicity of the graph."
    )
    prompt += "\n\n{{#user~}}\nThanks. Now please tell me whether the graph is\n-monotone increasing\n-monotone decreasing, or\n-not monotone.\n{{~/user}}\n\n"
    prompt += """{{#assistant~}}{{gen 'monotonicity' temperature=0.7 max_tokens=100}}{{~/assistant}}\n\n"""
    prompt += "\n\n{{#user~}}\nGreat. Now please summarize your previous answer by just replying with 'monotone increasing', 'monotone decreasing' or 'not monotone'.\n{{~/user}}\n\n"
    prompt += """{{#assistant~}}{{gen 'monotonicity_final' temperature=0.0 max_tokens=10}}{{~/assistant}}\n\n"""
    # run the prompts against the llm
    response = guidance(prompt, llm=llm)()
    # logging, but only if the folder for the logs exists
    log_folder = "../logs/monotonicity/gpt-4-0613"
    if os.path.exists(log_folder):
        # the current data, including the hour, minute and seconds
        now = datetime.now()
        current_time = now.strftime("%H-%M-%S")
        current_date = now.strftime("%d-%m-%Y")
        logfilename = os.path.join(
            log_folder, f"{current_date}-{current_time}-monotonicity.txt"
        )
        # log the response as a string
        with open(logfilename, "w") as logfile:
            logfile.write(str(response))
    return response["monotonicity_final"]


if __name__ == "__main__":
    # llm = test_util.openai_setup_gpt3_5()
    llm = test_util.openai_setup_gpt4()
    for dataset_name in test_util.get_avaialble_datasets():
        ebm = test_util.get_ebm(dataset_name)
        _, _, _, _, feature_names = test_util.get_dataset(dataset_name)
        response = None
        for feature_idx, feature_name in enumerate(feature_names):
            feature_graph = graphs.extract_graph(ebm, feature_idx)
            for graph in [feature_graph, invert_graph(feature_graph)]:
                graph = copy.deepcopy(graph)
                if graph.feature_type == "continuous":
                    graph = graphs.simplify_graph(graph, 0.05)
                    if is_monotone_increasing(graph):
                        response = decide_monotonicity(llm, graphs.graph_to_text(graph))
                        print(dataset_name, feature_name, "INCREASING", response)
                    elif is_monotone_decreasing(graph):
                        response = decide_monotonicity(llm, graphs.graph_to_text(graph))
                        print(dataset_name, feature_name, "DECREASING", response)
                    else:
                        response = decide_monotonicity(llm, graphs.graph_to_text(graph))
                        print(dataset_name, feature_name, "NONE", response)
                        pass
                    # sleep for 1 second to avoid rate limiting
                    time.sleep(1)
