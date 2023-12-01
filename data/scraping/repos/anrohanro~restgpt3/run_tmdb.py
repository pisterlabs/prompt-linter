import os
import json
import logging
import time
import yaml

from langchain.requests import Requests
from langchain.llms import OpenAI

from langchain.requests import Requests


from utils import reduce_openapi_spec, ColorPrint, MyRotatingFileHandler
from model import RestGPT

logger = logging.getLogger()

OPENAI_API_KEY= "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"
TMDB_ACCESS_TOKEN="Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJiZTc2ODMwZTQxZDA0ZmU3ZjUyOTY5YWQwOTRiM2Q5ZSIsInN1YiI6IjY1NWY3OTFkMmIxMTNkMDE0ZWFkNzhhNyIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.B8qRMknIxei3XT9rqOohWNqZt-AdMqDGc-XsTQF70Ks"


SPOTIPY_CLIENT_ID="4527f42d7b30476d81a71311df8e671d"
SPOTIPY_CLIENT_SECRET="f447f09a0234ec3b1feda9c19d6c461 "
SPOTIPY_REDIRECT_URI="https://github.com/Yifan-Song793/RestGPT/tree/main"


def run(query, api_spec, requests_wrapper, simple_parser=False):
    llm = OpenAI(model_name="text-davinci-003", temperature=0.0, max_tokens=256)
    # llm = OpenAI(model_name="gpt-3.5-turbo-0301", temperature=0.0, max_tokens=256)
    rest_gpt = RestGPT(llm, api_spec=api_spec, scenario='tmdb', requests_wrapper=requests_wrapper, simple_parser=simple_parser)

    logger.info(f"Query: {query}")

    start_time = time.time()
    rest_gpt.run(query)
    logger.info(f"Execution Time: {int(time.time() - start_time)} seconds")


def main():
    #config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
    #os.environ["OPENAI_API_KEY"] = config['openai_api_key']
    #os.environ["TMDB_ACCESS_TOKEN"] = config['tmdb_access_token']

    log_dir = os.path.join("logs", "restgpt_tmdb")
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    file_handler = MyRotatingFileHandler(os.path.join(log_dir, f"tmdb.log"), encoding='utf-8')
    logging.basicConfig(
        format="%(message)s",
        handlers=[logging.StreamHandler(ColorPrint()), file_handler],
    )
    logger.setLevel(logging.INFO)

    with open("specs/tmdb_oas.json") as f:
        raw_tmdb_api_spec = json.load(f)

    api_spec = reduce_openapi_spec(raw_tmdb_api_spec, only_required=False)

    access_token = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJiZTc2ODMwZTQxZDA0ZmU3ZjUyOTY5YWQwOTRiM2Q5ZSIsInN1YiI6IjY1NWY3OTFkMmIxMTNkMDE0ZWFkNzhhNyIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.B8qRMknIxei3XT9rqOohWNqZt-AdMqDGc-XsTQF70Ks"

    headers = {
        'Authorization': f'Bearer {access_token}'
    }

    requests_wrapper = Requests(headers=headers)

    queries = json.load(open('datasets/tmdb.json', 'r'))
    queries = [item['query'] for item in queries]

    for idx, query in enumerate(queries, 1):
        try:
            print('#' * 20 + f" Query-{idx} " + '#' * 20)
            run(query, api_spec, requests_wrapper, simple_parser=False)
        except Exception as e:
            print(f"Query: {query}\nError: {e}")
        finally:
            file_handler.doRollover()
    

if __name__ == '__main__':
    main()
