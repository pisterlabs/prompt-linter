import json
import logging
import os
import sys
from dotenv import load_dotenv
from tqdm import tqdm

from openai_client import OpenAIClient
from retriever import Retriever
from sim_caculator import SimCaculator


if __name__ == "__main__":
    start_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    data_file_path = "./eval_data/filtered/data2.jsonl"
    sum_result_root_path = "./eval_data/sum_result"
    ret_log_dir_path = "./eval_data/ret_log"
    ret_result_file_path = "./eval_data/ret_result.jsonl"

    load_dotenv()  # load environment variables from .env file

    logging.basicConfig(level=logging.INFO,
                        format='%(name)s - %(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S')
    pipeline_logger = logging.getLogger("pipeline")

    if not os.path.exists(ret_log_dir_path):
        os.mkdir(ret_log_dir_path)

    # create client for OpenAI
    try:
        openai_client = OpenAIClient()
    except Exception as e:
        pipeline_logger.error(e)
        exit(1)
    # check if enough credits
    # if openai_client.get_credit_grants() < 2.0:
    #     pipeline_logger.error("Not enough credits to retrieval.")
    #     exit(1)

    # create similarity caculator
    sim_calculator = SimCaculator()

    with open(data_file_path, "r") as f_data, open(ret_result_file_path, "a") as f_ret_result:
        data_objs = [json.loads(line) for line in f_data.readlines()]

        for idx, data_obj in enumerate(tqdm(data_objs[start_idx:])):
            try:
                query = data_obj['query']
                repo_name = data_obj['repo'].split('/')[-1]

                sum_result_dir_path = os.path.join(
                    sum_result_root_path, repo_name)

                sum_out_path = os.path.join(
                    sum_result_dir_path, f"sum_out_{repo_name}.json")
                ret_log_path = os.path.join(
                    ret_log_dir_path, f"ret_log_{data_obj['id']}.txt")

                # check if existence of path
                if not os.path.exists(sum_out_path):
                    raise Exception("Summary output path does not exist.")

                # create loggers
                ret_logger = logging.getLogger(ret_log_path)
                ret_logger.addHandler(
                    logging.FileHandler(ret_log_path, "w", "utf-8")
                )
                ret_logger.propagate = False  # prevent printing to console

                # retrieve method according to the description
                retriever = Retriever(
                    ret_logger, openai_client, sim_calculator)
                with open(sum_out_path, "r") as f_sum_out:
                    repo_sum_obj = json.loads(f_sum_out.read())
                    res_obj = retriever.retrieve_in_repo(query, repo_sum_obj)

                    if res_obj['is_error']:
                        raise Exception("An error occurred during retrieval.")

                    res_obj['id'] = data_obj['id']

                    # write result to file
                    f_ret_result.write(json.dumps(res_obj) + '\n')
                    f_ret_result.flush()

            except Exception as e:
                pipeline_logger.error(e)
                pipeline_logger.warning(f'Stop at {idx + start_idx}')
                break

    logging.shutdown()
