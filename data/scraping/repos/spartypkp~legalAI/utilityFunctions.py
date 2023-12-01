import tiktoken
import openai
import json
import config
import os
from tenacity import retry, wait_random_exponential, stop_after_attempt
import psycopg2
from config import config_psql
import time
import asyncio
import aiohttp
DIR = os.path.dirname(os.path.realpath(__file__))
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {config.spartypkp_openai_key}"
}

#start_time = time.perf_counter()
#print("    * Time elapsed: ", round(time.perf_counter() - start_time, 3), "seconds.")
openai.api_key = config.spartypkp_openai_key
codes = ["BPC","CCP","CIV","COM","CONS","CORP","EDC","ELEC","EVID","FAC","FAM","FGC","FIN","GOV","HNC","HSC","INS","LAB","MVC","PCC","PEN","PRC","PROB","PUC","RTC","SHC","UIC","VEH","WAT","WIC"]

def main():
    pass
    # DENOTES NEW SECTION
    # \u00a0\u00a0

## DECORATORS
# Debug decorator specifically for gpt completions
def gpt_wrapper(func):
    def inner(*args, **kwargs):
        if "debug_print" in kwargs:
            print_debug = kwargs["debug_print"]
        else:
            print_debug = False
        #if print_debug:
            #print("## Before openAI create_chat_completion:\n## Used Model: {}, API Key: {}\n## Function Input: {}".format(kwargs["used_model"], kwargs["api_key_choice"], kwargs["prompt_messages"]))
        begin = time.time()
        returned_value = func(*args, **kwargs)
        end = time.time()
        if print_debug:
            prompt_tokens = returned_value.usage["prompt_tokens"]
            completion_tokens = returned_value.usage["completion_tokens"]
            total_tokens = returned_value.usage["total_tokens"]
            total_cost = calculate_prompt_cost(kwargs["used_model"], prompt_tokens, completion_tokens)
            print("    * Total time in {}: {}, Total Tokens: {}, Total Cost: ${}".format(func.__name__, round(end-begin, 2), total_tokens, round(total_cost, 2)))
        return returned_value
    return inner

# General debug decorator
def debug(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__} with args: {args}, \
        kwargs: {kwargs}")
        begin = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print("{} ran in {}.".format(func.__name__, end-begin))
        print(f"{func.__name__} returned: {result}")
        return result

    return wrapper


# Deprecated
def find_big_tokens():
    bigTokens = {}
    for code in codes:
        with open("{}.txt".format(code), 'r') as convert_file:
            rawText = convert_file.read()
            text = json.loads(rawText)

        for k,v in text.items():
            splt = v.split(",")
            numTokens = int(splt[0])
            if numTokens > 7000:
                print("Key {} has BIGTOKENS {}".format(k, numTokens))
                bigTokens[k] = numTokens
    print("Number of bigtoken offenders:", len(bigTokens.keys()))
    with open("bigtokens.txt", 'w') as out_file:
        out_file.write(json.dumps(bigTokens))
    out_file.close

def num_tokens_from_string(string):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Create embeddings, previously embedCodes.py
def get_embedding_and_token(text, model="text-embedding-ada-002"):
    embed = openai.Embedding.create(input=[text], model=model)
    return embed["data"][0]["embedding"], embed["usage"]["total_tokens"]

# Return just the embedding
@retry(wait=wait_random_exponential(min=1, max=2), stop=stop_after_attempt(6))
def get_embedding(text, model="text-embedding-ada-002"):
    return openai.Embedding.create(input=[text],model=model)["data"][0]["embedding"]
    
# PSQL Access Functions, previously getPSQLConn.py
def psql_connect():
    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        # read connection parameters
        params = config_psql()

        # connect to the PostgreSQL server
        # print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)
		
        return conn
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        raise error

def select_and_fetch_rows(conn, sql_select):
    cursor = conn.cursor()
    cursor.execute(sql_select)
    rows = cursor.fetchall()
    cursor.close()
    return rows


@gpt_wrapper
@retry(wait=wait_random_exponential(min=1, max=2), stop=stop_after_attempt(6))
def create_chat_completion(used_model="gpt-3.5-turbo", api_key_choice="will", prompt_messages=None, debug_print=False, temp=0.4, top_p_val=1, n_responses=1, do_stream=False, stop_conditions=None, presence_p=0, frequency_p=0):
    openai.api_key = config.get_api_key(api_key_choice)
    try:
        chat_completion = openai.ChatCompletion.create(model=used_model,messages=prompt_messages, temperature=temp, n=n_responses, top_p=top_p_val, stream=do_stream, stop=stop_conditions, presence_penalty=presence_p, frequency_penalty=frequency_p)
        return chat_completion
    except Exception as e:
        print("***** Failed calling create_chat_completion!")
        print(e)
        raise e

@retry(wait=wait_random_exponential(min=1, max=2), stop=stop_after_attempt(6))
def stream_chat_completion(used_model="gpt-3.5-turbo", api_key_choice="will", prompt_messages=None, temp=0.4, top_p_val=1, n_responses=1, do_stream=True, stop_conditions=None, presence_p=0, frequency_p=0):
    openai.api_key = config.get_api_key(api_key_choice)
    try:
        chat_completion = openai.ChatCompletion.create(model=used_model,messages=prompt_messages, temperature=temp, n=n_responses, top_p=top_p_val, stream=do_stream, stop=stop_conditions, presence_penalty=presence_p, frequency_penalty=frequency_p)
        for message in chat_completion:
            yield message
    except Exception as e:
        print("***** Failed calling create_chat_completion!")
        print(e)
        raise e

# Prompt cost calculations
def calculate_prompt_cost(model, prompt_tokens, completion_tokens):
    model_rates = {"gpt-3.5-turbo-16k":[0.003, 0.004], "gpt-3.5-turbo":[0.0015, 0.002], "gpt-4":[0.03, 0.06], "gpt-4-32k":[0.06, 0.12]}
    prompt_rate = model_rates[model][0]
    completion_rate = model_rates[model][1]
    cost = ((prompt_rate/1000)*prompt_tokens) + ((completion_rate/1000)*completion_tokens)
    #print("Prompt Tokens: {}, Completion Tokens: {}".format(prompt_tokens, completion_tokens))
    #print("Total cost of using {}: ${}".format(model, cost))
    return cost

class ProgressLog:
    def __init__(self, total, model):
        self.total = total
        self.done = 0
        self.model = model

    def increment(self):
        self.done = self.done + 1

    def __repr__(self):
        return f"    * OpenAI {self.model} API call {self.done}/{self.total}."

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(20), before_sleep=print, retry_error_callback=lambda _: None)
async def get_completion(content, session, semaphore, progress_log, used_model):
    async with semaphore:

        async with session.post("https://api.openai.com/v1/chat/completions", headers=HEADERS, json={
            "model": used_model,
            "messages": content,
            "temperature": 0
        }) as resp:

            response_json = await resp.json()

            progress_log.increment()
            print(progress_log)

            return response_json

async def get_completion_list(content_list, max_parallel_calls, used_model="gpt-3.5-turbo"):
    semaphore = asyncio.Semaphore(value=max_parallel_calls)
    progress_log = ProgressLog(len(content_list), used_model)

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(10)) as session:
        return await asyncio.gather(*[get_completion(content, session, semaphore, progress_log, used_model) for content in content_list])







if __name__ == "__main__":
    main()