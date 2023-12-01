import dataclasses
import logging
import math
import os
import io
import sys
import time
import json
from typing import Optional, Sequence, Union

import openai
import tqdm
from openai import openai_object
import copy
import redis
import requests

import anthropic
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_not_exception_type


claude = anthropic.Anthropic()

StrOrOpenAIObject = Union[str, openai_object.OpenAIObject]

openai_org = os.getenv("OPENAI_ORG")
if openai_org is not None:
    openai.organization = openai_org
    logging.warning(f"Switching to organization: {openai_org} for OAI API key.")


@dataclasses.dataclass
class OpenAIDecodingArguments(object):
    max_tokens: int = 1800
    temperature: float = 0.2
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Sequence[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    suffix: Optional[str] = None
    logprobs: Optional[int] = None
    echo: bool = False

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def claude_gpt_retry(prompt: str, model_name="claude-2", max_tokens_to_sample = 6000) -> str:
    stream = claude.completions.create(
        prompt=prompt,
        stop_sequences=[anthropic.HUMAN_PROMPT],
        model=model_name,
        stream=True,
        max_tokens_to_sample=max_tokens_to_sample,
    )

    result = ""
    for completion in stream:
        result += completion.completion
    # print(f"********************** Chat Response **********************\n\n{result}")

    return result

def claude_gpt(prompt: str, model_name="claude-2", max_tokens_to_sample = 6000) -> str:
    """
    This function sends a prompt to the Anthropic's Claude API and returns the response.
    
    Args:
        prompt (str): Prompt to send to the API.
        

    Returns:
        str: The response from the API.
    """
    # send the prompt to gpt and return the response
    try:
        return claude_gpt_retry(prompt, model_name, max_tokens_to_sample)    
    except Exception as e:
        print(f"Error occured invoking Claude: {e}")
        return None

RETRIEVAL_API = 'http://34.204.174.75:8080/search'
EMBEDDING_API = 'http://34.204.174.75:9000/embedding'

# change this to remote
redis_client = redis.Redis(host='34.204.174.75', port=6379, decode_responses=True)

def get_embedding(text):
    request = {
        'message': text,
        'instruction': 'Represent this sentence for searching relevant passages: '
    }
    return requests.post(EMBEDDING_API, json = request).json()["embedding"]

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_context(question, context_count=5):
    embedding = get_embedding(question)
    
    request = {
        'vector': embedding,
        'num_results': context_count
    }

    payload = requests.post(RETRIEVAL_API, json = request).json()
    vector_raw_ids = payload['ids']
    similarities = payload['dists']
    logging.info(f'question: {question}\ndocIds: {vector_raw_ids}\nsimilarities: {similarities}')

    keys = []
    documents = []

    for i in range(len(vector_raw_ids)):
        if similarities[i] > .5:
          keys.append(vector_raw_ids[i])

    if len(keys) > 0:
        documents = redis_client.mget(keys)

    doc_map = {}

    for i in range(len(keys)):
        if documents[i]:
          doc = json.loads(documents[i])         
          doc_map[keys[i]] = {
            'content': doc['content'],
            'source': doc['source'],
            'title': doc['title'],
            'page': doc['page'],
            'link': doc['link'] + "#page=" + str(doc['page'])
          }
    
    sources_map = {}
    context = ''
    context_log = 'CONTEXT\n\n'
    
    for i in range(len(vector_raw_ids)):
        id = vector_raw_ids[i]
        if id in doc_map:
            doc = doc_map[id]
            context_log += f'id: {id}\nsimilarity:{similarities[i]}\n{doc["link"]}\n\n'
            content = doc['content'].strip()
            ## Remove the first line of the page #
            # first_new_line_index = max(content.find('\n'), 200)
            # content = content[first_new_line_index:]
            ######################################
            sources_map[id] = {
                'id': id,
                'title': doc['title'],
                'linkLabel': "Page " + str(doc['page']),
                'link': doc['link']
            }
            
            context += f"""

<context>
  {content}
  <id>{id}</id>
</context>"""

        else:
            context_log += f'id: {id}\nsimilarity:{similarities[i]} <dropped>\n\n'            

    return {
        'sources_map': sources_map,
        'context': context,
        'context_log': context_log
    }


def openai_gpt(prompt: str, model_name='gpt-3.5-turbo', max_attempts: int = 3) -> str:
    """
    This function sends a prompt to the OpenAI GPT API and returns the response.
    It tries the creation several times (max_attempts) in case of exception.
    If the model is text-davinci-003, it uses the Completion API, otherwise it uses the ChatCompletion API.

    Args:
        prompt (str): Prompt to send to the API.
        

    Returns:
        str: The response from the API.
    """
    # send the prompt to gpt and return the response
    # try the creation several times in case of exception
    for attempt in range(1, max_attempts + 1):
        try:
            messages = [
                # {"role": "system", "content": ""},
                {"role": "user", "content": f"{prompt}"},
            ]
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=messages,
                # max_tokens=3072,
                # temperature=0.5,
                # top_p=1.0,
                # stop=["\n20", "20.", "20."]
            )
            choices = [choice["message"]["content"] for choice in response["choices"]]

            print(f"********************** Chat Response **********************\n\n{choices[0]}")

            return choices[0]
        except openai.error.OpenAIError as e:
            if attempt < max_attempts:
                print(f"Error on attempt {attempt}: {e}. Retrying...")
                time.sleep(2)  # Wait for 2 seconds before retrying
            else:
                print(f"Error on attempt {attempt}: {e}. All attempts failed.")
                # we will return None if all attempts failed because raising an exception will stop the program and we will lose all the data we have collected so far
                return None


def openai_completion(
    prompts: Union[str, Sequence[str], Sequence[dict[str, str]], dict[str, str]],
    decoding_args: OpenAIDecodingArguments,
    model_name="text-davinci-003",
    sleep_time=2,
    batch_size=1,
    max_instances=sys.maxsize,
    max_batches=sys.maxsize,
    return_text=False,
    **decoding_kwargs,
) -> Union[Union[StrOrOpenAIObject], Sequence[StrOrOpenAIObject], Sequence[Sequence[StrOrOpenAIObject]],]:
    """Decode with OpenAI API.

    Args:
        prompts: A string or a list of strings to complete. If it is a chat model the strings should be formatted
            as explained here: https://github.com/openai/openai-python/blob/main/chatml.md. If it is a chat model
            it can also be a dictionary (or list thereof) as explained here:
            https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb
        decoding_args: Decoding arguments.
        model_name: Model name. Can be either in the format of "org/model" or just "model".
        sleep_time: Time to sleep once the rate-limit is hit.
        batch_size: Number of prompts to send in a single request. Only for non chat model.
        max_instances: Maximum number of prompts to decode.
        max_batches: Maximum number of batches to decode. This argument will be deprecated in the future.
        return_text: If True, return text instead of full completion object (which contains things like logprob).
        decoding_kwargs: Additional decoding arguments. Pass in `best_of` and `logit_bias` if you need them.

    Returns:
        A completion or a list of completions.
        Depending on return_text, return_openai_object, and decoding_args.n, the completion type can be one of
            - a string (if return_text is True)
            - an openai_object.OpenAIObject object (if return_text is False)
            - a list of objects of the above types (if decoding_args.n > 1)
    """
    is_single_prompt = isinstance(prompts, (str, dict))
    if is_single_prompt:
        prompts = [prompts]

    if max_batches < sys.maxsize:
        logging.warning(
            "`max_batches` will be deprecated in the future, please use `max_instances` instead."
            "Setting `max_instances` to `max_batches * batch_size` for now."
        )
        max_instances = max_batches * batch_size

    prompts = prompts[:max_instances]
    num_prompts = len(prompts)
    prompt_batches = [
        prompts[batch_id * batch_size : (batch_id + 1) * batch_size]
        for batch_id in range(int(math.ceil(num_prompts / batch_size)))
    ]

    completions = []
    for batch_id, prompt_batch in tqdm.tqdm(
        enumerate(prompt_batches),
        desc="prompt_batches",
        total=len(prompt_batches),
    ):
        batch_decoding_args = copy.deepcopy(decoding_args)  # cloning the decoding_args

        while True:
            try:
                shared_kwargs = dict(
                    model=model_name,
                    **batch_decoding_args.__dict__,
                    **decoding_kwargs,
                )
                completion_batch = openai.Completion.create(prompt=prompt_batch, **shared_kwargs)
                choices = completion_batch.choices

                for choice in choices:
                    choice["total_tokens"] = completion_batch.usage.total_tokens
                completions.extend(choices)
                break
            except openai.error.OpenAIError as e:
                logging.warning(f"OpenAIError: {e}.")
                if "Please reduce your prompt" in str(e):
                    batch_decoding_args.max_tokens = int(batch_decoding_args.max_tokens * 0.8)
                    logging.warning(f"Reducing target length to {batch_decoding_args.max_tokens}, Retrying...")
                else:
                    logging.warning("Hit request rate limit; retrying...")
                    time.sleep(sleep_time)  # Annoying rate limit on requests.

    if return_text:
        completions = [completion.text for completion in completions]
    if decoding_args.n > 1:
        # make completions a nested list, where each entry is a consecutive decoding_args.n of original entries.
        completions = [completions[i : i + decoding_args.n] for i in range(0, len(completions), decoding_args.n)]
    if is_single_prompt:
        # Return non-tuple if only 1 input and 1 generation.
        (completions,) = completions
    return completions


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict
