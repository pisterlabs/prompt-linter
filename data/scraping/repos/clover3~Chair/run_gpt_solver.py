import logging
import sys
import time
2
import openai
from openai.error import RateLimitError, ServiceUnavailableError, OpenAIError

from taskman_client.wrapper3 import JobContext
from utils.open_ai_api import ENGINE_GPT4, ENGINE_GPT_3_5
from dataset_specific.scientsbank.eval_helper import solve_eval_report
from dataset_specific.scientsbank.parse_fns import get_split_spec, load_scientsbank_split
from dataset_specific.scientsbank.pte_data_types import PTEPredictionPerQuestion, Question
from dataset_specific.scientsbank.pte_solver_if import apply_solver
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.tli.pte.gpt_solver import get_gpt_requester, get_gpt_read_solver
from typing import List, Iterable, Callable, Dict, Tuple, Set


def apply_solver_loop(requester, questions):
    n_fail = 0
    is_success = False
    while n_fail < 1000:
        try:
            apply_solver(requester, questions)
            is_success = True
            break
        except OpenAIError as e:
            print(e)
            print(e.json_body)
            n_fail += 1
            wait_time = 5
            c_log.info("Encountered server error retry in %d seconds", wait_time)
            time.sleep(wait_time)
    return is_success


def solve_for_split(split_name):
    run_name = "gpt_{}".format(split_name)
    with JobContext(run_name):
        c_log.setLevel(logging.DEBUG)
        split = get_split_spec(split_name)
        engine = ENGINE_GPT_3_5
        questions: List[Question] = load_scientsbank_split(split)
        c_log.info("Building solver")
        requester = get_gpt_requester(engine, split_name)
        c_log.info("Running with loop")
        is_success = apply_solver_loop(requester, questions)
        if not is_success:
            c_log.error("Maximum 10 iteration reached")
            return
        solver = get_gpt_read_solver(engine, split_name)
        solve_eval_report(solver, split)


def main():
    split_name = sys.argv[1]
    solve_for_split(split_name)


if __name__ == "__main__":
    main()