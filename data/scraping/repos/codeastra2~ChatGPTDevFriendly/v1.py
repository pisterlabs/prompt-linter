import openai
import time
from functools import wraps
import requests
import logging
import sys
import json

import matplotlib.pyplot as plt
from datetime import datetime


log_formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s", "%m-%d-%Y %H:%M:%S"
)


def retry(tries: int = 5, delay: int = 3, backoff: int = 2):
    """Retry  wrapper, to retry functions on exceptions and errors.

    Args:
        tries (int, optional): Number of retries. Defaults to 5.
        delay (int, optional): Time delay in seconds. Defaults to 3.
        backoff (int, optional): Backoff between the retries. Defaults to 2.
    """

    def deco_retry(func):
        @wraps(func)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries >= 0:
                try:
                    # args[0].logger.info("Trying the the function.......")
                    return func(*args, **kwargs)
                except (
                    openai.error.APIError,
                    openai.error.APIConnectionError,
                    requests.exceptions.RequestException,
                ) as e:
                    args[0].logger.info(f"Error occurred: {str(e)}")
                    mtries -= 1
                    if mtries >= 0:
                        args[0].logger.info(f"Retrying after {mdelay} seconds...")
                        time.sleep(mdelay)

                    mdelay *= backoff
            return func(*args, **kwargs)

        return f_retry

    return deco_retry


def dot_product(list1: list, list2: list):
    """Computes the dot product of 2 vectors.

    Args:
        list1 (int): List 1 values
        list2 (int): List 2 values

    Raises:
        ValueError: _description_

    Returns:
        int: Dot product value of the 2 lists.
    """
    # Check if the lengths of the two lists are equal
    if len(list1) != len(list2):
        raise ValueError("Lists must have same length")

    # Calculate the dot product using a loop
    dot_product = 0
    for i in range(len(list1)):
        dot_product += list1[i] * list2[i]

    return dot_product


class ChatGptSmartClient(object):
    """
    This is a wrapper class for the chatgpt python api,
    it is meant to provide developers a smooth expereince in
    developing chatgpt applications of their own without the need
    for worrying about things like retries, tracking message history
    or storing messages. This runs on top of the Chat APIs provided by OpenAI
    read more on that here: https://platform.openai.com/docs/guides/chat .
    """

    CONTEXT_TOKEN_LIMIT = 4000

    def __init__(self, api_key: str, model: str, log_info: bool = False):
        """Init method to instantiate the client.

        Args:
            api_key (str): OpenAI api key.
            model (str): The model to be used for chat completion, ex: "gpt-3.5-turbo"
            log_info (bool, optional): Whether to log stats for each query. Defaults to False.
        """
        openai.api_key = api_key

        self.instruction_msgs = {
            "role": "system",
            "content": "You are a  useful assistant",
        }
        self.prev_msgs = [self.instruction_msgs]
        self.model = model
        self.rsp_id = 1
        self.query_id = 0

        self.rsp_time_list = []
        self.rsp_tstamp_list = []
        self.total_token_cnt_list = []

        self.rspid_vs_tottokens_dict = {}

        self.total_token_cnt = 0

        self.log_info = log_info
        self.logger = logging.getLogger(f"chatgptlogger{time.time()}")
        self.logger.setLevel(logging.INFO)
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(log_formatter)
        self.logger.addHandler(stdout_handler)

    @retry()
    def query(self, query: str, w_context: bool = True, add_to_context: bool = True):
        """Wrapper method to make conversation with ChatGPT apis.

        Args:
            query (str): Query to send to Chatgpt servers.
            w_context (bool, optional): Whether to ask this query with previous conversation context. Defaults to True.
            add_to_context (bool, optional): Whether to add this query and response to conversation context. Defaults to True.

        Returns:
            OpenAIJson, int: Response to query  and the response id for later rollbck if necessary
        """

        # TODO: We coud get the embeddings, cache and further use them to speed up the results.
        # self.get_embeddings(query=query)

        query = {"role": "user", "content": query}
        rsp_id = None

        if sum(self.total_token_cnt_list) >= self.CONTEXT_TOKEN_LIMIT:
            self.trim_conversation()

        if w_context:
            msgs = self.prev_msgs[:]
            msgs.append(query)
        else:
            msgs = [self.instruction_msgs, query]

        start_time = time.time()
        response = openai.ChatCompletion.create(model=self.model, messages=msgs)
        end_time = time.time()

        if self.log_info:
            self.logger.info(
                f"The time taken for this response is : {end_time - start_time} seconds"
            )
        # print(f"The time taken for this response is : {end_time - start_time} seconds")

        f_resp = response["choices"][0]["message"]
        tot_token_cnt = response["usage"]["total_tokens"]

        self.rsp_tstamp_list.append(end_time)
        self.rsp_time_list.append(end_time - start_time)
        self.total_token_cnt_list.append(tot_token_cnt)

        self.total_token_cnt += tot_token_cnt

        if self.log_info:
            self.logger.info(
                f"The total token count currently is {self.total_token_cnt}"
            )

        if add_to_context:
            self.prev_msgs.append(query)
            self.prev_msgs.append(f_resp)
            self.rsp_id += 1
            rsp_id = self.rsp_id
            self.rspid_vs_tottokens_dict[self.rsp_id] = tot_token_cnt

        # print(self.prev_msgs)

        return f_resp, rsp_id

    def erase_history(self) -> None:
        """Removing all previous context."""
        self.prev_msgs = [self.instruction_msgs]
        self.rsp_id = len(self.prev_msgs)

    # This function is used for getting embeddings and hence maybe
    # used to speedup the system by caching.
    def get_embeddings(self, query: str) -> None:
        """Calls the OpenAI embeddings API to get the generated embeddings for a query,
            can be used later for caching and faster response time.

        Args:
            query (str): Query to be sent to OpenAI servers Chat Apis.
        """
        response = openai.Embedding.create(input=query, model="text-embedding-ada-002")
        embeddings = response["data"][0]["embedding"]
        return embeddings

    def rollback_conversation(self, rsp_id: int) -> None:
        """Rollback conversation to the point of a particular response.

        Args:
            rsp_id (int): Id number of previous tracked response to roll back to.
        """
        self.prev_msgs = self.prev_msgs[0:rsp_id]
        self.rsp_id = len(self.prev_msgs)

    def print_metrics(self) -> None:
        """Method to log the token usage and response time information."""
        self.logger.info(
            f"The total tokens used up-till now is: {self.total_token_cnt}"
        )
        self.logger.info(
            f"The average response time is: {sum(self.rsp_time_list)/len(self.rsp_time_list)} sec"
        )

        self.plot_rsp_times()

    def plot_rsp_times(self) -> None:
        """Method to plot the response times versus the timestamps."""
        formatted_timestamps = [
            datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
            for ts in self.rsp_tstamp_list
        ]

        # Plot the response times against the timestamps
        fig, ax = plt.subplots()
        ax.plot(formatted_timestamps, self.rsp_time_list)
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Response Time (s)")
        ax.set_title("ChatGPT API Response Time")
        ax.tick_params(axis="x", rotation=45)
        ax.xaxis.labelpad = 85
        ax.yaxis.labelpad = 35

        # Increase the bottom and left margins of the plot
        plt.subplots_adjust(bottom=0.2, left=0.15)

        plt.xticks(rotation=45, fontsize=6)

        # Save the figure as a PNG file
        plt.savefig("response_times.png")

        # Show the plot
        plt.show()

    def dump_context_to_a_file(self, filename: str = "context"):
        """Wrapper method to call to dump context to a file.

        Args:
            filename (str, optional): Filename to dump to. Defaults to "context".
        """
        self.dicts_to_jsonl(data_list=self.prev_msgs, filename=filename)

    def dicts_to_jsonl(self, data_list: list, filename: str) -> None:
        """Method saves context list of dicts to .jsonl format.

        Args:
            data_list (list): List of dicts of previous conversations
            filename (str): Filenam to save to
        """
        sjsonl = ".jsonl"

        # Check filename
        if not filename.endswith(sjsonl):
            filename = filename + sjsonl

        # Save data
        with open(filename, "w") as out:
            for ddict in data_list:
                jout = json.dumps(ddict) + "\n"
                out.write(jout)

    def load_context_from_a_file(self, filename: str):
        """Fills up the self.prev_msgs i.e context from a dumped file.

        Args:
            filename (str): Filename should be a .jsonl file.

        Returns:
            list: List containing the context information
        """
        sjsonl = ".jsonl"

        # Check filename
        if not filename.endswith(sjsonl):
            filename = filename + sjsonl

        self.prev_msgs = []

        with open(filename, encoding="utf-8") as json_file:
            for line in json_file.readlines():
                self.prev_msgs.append(line)

        return self.prev_msgs

    def trim_conversation(self):
        """Method to trim, context generatd till now to below the token limit.
        The queries are removed in FIFO manner until the token count goes below the limit.
        """
        while sum(self.total_token_cnt_list) >= self.CONTEXT_TOKEN_LIMIT:
            del self.total_token_cnt_list[0]
            del self.prev_msgs[1]
            del self.prev_msgs[2]

        self.logger.info(
            f"Trimmed the context list to length: {sum(self.total_token_cnt_list)}"
        )
