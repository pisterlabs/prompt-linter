import os
import re
import hashlib
from datetime import datetime

from langchain.text_splitter import (
    TextSplitter,
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    NLTKTextSplitter,
    SpacyTextSplitter,
    TokenTextSplitter,
)


@staticmethod
def get_default_text_splitter(method) -> TextSplitter:
    # Note on different chunking strategies https://www.pinecone.io/learn/chunking-strategies/
    # Note that RecursiveCharacterTextSplitter can currently enter infinite loop:
    # see https://github.com/hwchase17/langchain/issues/1663
    method = method.lower()
    if method == "character":
        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    elif method == "recursive":
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    elif method == "nltk":
        text_splitter = NLTKTextSplitter(chunk_size=1000)
    elif method == "spacy":
        text_splitter = SpacyTextSplitter(chunk_size=1000)
    elif method == "tiktoken":
        text_splitter = TokenTextSplitter(chunk_size=10, chunk_overlap=0)
    else:
        raise ValueError(f"argument method {method} is not supported")
    return text_splitter


@staticmethod
def get_secrets(key_name):
    _key_path = f"secrets/{key_name}.key"
    if os.path.exists(_key_path):
        _key_file = open(_key_path, "r", encoding="utf-8")
        _key_value = _key_file.read()
    else:
        _key_value = None
    return _key_value


@staticmethod
def get_word_match_list(text, word_list):
    joined_words = "|".join(word_list)
    match_list = re.findall(joined_words, text, flags=re.IGNORECASE)
    match_list = [i.lower() for i in match_list]
    return match_list


@staticmethod
def get_epoch_time():
    return (datetime.now() - datetime(1970, 1, 1)).total_seconds()


class agent_logs:
    # add log state to allow for caching of logs

    @classmethod
    def set_cache_lookup(cls, text):
        # set the lookup value for the cache, usually the main promopt plus the agent type
        hasher = hashlib.md5(text.strip().encode())
        cls.cache_hash = hasher.hexdigest()
        cls.cache_file = f"logs/saved/output_{cls.cache_hash}.log"
        # try to load from saved cache
        if os.path.exists(cls.cache_file):
            cache_log = cls.read_log(cls.cache_file)
            # overwrite current log with cache log
            with open("logs/output_now.log", "w") as f:
                print(cache_log, file=f)
            if len(cache_log.split("Final Answer:")) == 1:
                # no final answer provided, return the last observation action pair
                return cache_log.split("Thought:")[-1].strip()
            elif len(cache_log.split("Final Answer:")) > 1:
                # has final answer return it
                return cache_log.split("Final Answer:")[-1].strip()
            else:
                split_log = cache_log.split("\n")
                if len(split_log) == 0:
                    return None
                last_n_paragraphs = (5 if len(split_log) > 5 else len(split_log)) * -1
                # not handled, so return the last 5 paragraphs
                return "\n".join(split_log[-last_n_paragraphs:-1])
        return None

    @classmethod
    def save_cache(cls):
        # save the current cache
        try:
            # read current log
            current_log = cls.read_log()
            # save another copy of the log to cache log file
            with open(cls.cache_file, "w") as f:
                print(current_log, file=f)
        except:
            print("cannot save cache log if cache lookup is not defined")

    @staticmethod
    def write_log_and_print(text, ans_type=None):
        processed_text = agent_logs.write_log(text, ans_type)
        print(processed_text)

    @staticmethod
    def write_log(text, ans_type=None):
        # check if final answer
        if ans_type == "answer":
            text = f"Answer: {text}"
        elif ans_type == "final":
            text = f"Final Answer: {text}"
        # clean up alpaca style prompts
        text = text.replace(
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n",
            "",
        )
        text = text.replace(
            "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n",
            "",
        )
        text = text.replace(
            "Prompt after formatting:\n",
            "",
        )
        text = text.replace(
            "### Instruction:\n",
            "Query: ",
        )
        text = text.replace(
            "### Input:\n",
            "",
        )
        text = text.replace(
            "### Response:\n",
            "",
        )
        # clean up extra characters
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        text_to_log = re.sub(ansi_escape, "", text)
        text_to_log = re.sub(r"[\xc2\x99]", "", text_to_log).strip()
        # save output
        with open("logs/output_now.log", "a") as f:
            print(text_to_log, file=f)
        if os.getenv("MYLANGCHAIN_SAVE_CHAT_HISTORY") == "1":
            with open("logs/output_recent.log", "a") as f:
                print(f"======\n{text_to_log}\n", file=f)
        return text_to_log

    @staticmethod
    def read_log(log_file="logs/output_now.log"):
        # optioanlly, read external log output from langchain
        # require modifying packages/langchain/langchain/input.py
        with open(log_file, "r", encoding="utf-8") as f:
            current_log = f.read()
            return current_log

    @staticmethod
    def clear_log(log_file="logs/output_now.log"):
        # clear log so previous results don't get displayed
        with open(log_file, "w") as f:
            print("", file=f)
