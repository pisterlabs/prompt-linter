"""Script to run end-to-end evaluation on the benchmark"""
import argparse
import glob
import json
import logging
import os
import random
import subprocess
import tempfile
import time
from pathlib import Path

import pandas as pd
import openai

from agent import (
    Agent,
    PromptAgent,
    TeacherForcingAgent,
    construct_agent,
    LCAgent,
)
from agent.prompts import *
from browser_env import (
    Action,
    ActionTypes,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    create_stop_action,
)
from browser_env.actions import is_equivalent
from browser_env.auto_login import get_site_comb_from_filepath
from browser_env.helper_functions import (
    RenderHelper,
    get_action_description,
)
from evaluation_harness import evaluator_router

import my_globals
from pydantic.v1 import ValidationError

LOG_FOLDER = "log_files"
Path(LOG_FOLDER).mkdir(parents=True, exist_ok=True)
LOG_SUFFIX = f"{time.strftime('%Y%m%d%H%M%S', time.localtime())}_{random.randint(0, 10000)}"
LOG_FILE_NAME = f"{LOG_FOLDER}/log_{LOG_SUFFIX}.log"

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(LOG_FILE_NAME)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

# Set the log format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render the browser"
    )
    parser.add_argument(
        "--slow_mo",
        type=int,
        default=0,
        help="Slow down the browser by the specified amount",
    )
    parser.add_argument(
        "--action_set_tag", default="id_accessibility_tree", help="Action type"
    )
    parser.add_argument(
        "--observation_type",
        choices=["accessibility_tree", "html", "image"],
        default="accessibility_tree",
        help="Observation type",
    )
    parser.add_argument(
        "--current_viewport_only",
        action="store_true",
        help="Only use the current viewport for the observation",
    )
    parser.add_argument("--viewport_width", type=int, default=1280)
    parser.add_argument("--viewport_height", type=int, default=720)
    parser.add_argument("--save_trace_enabled", action="store_true")
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)

    parser.add_argument("--max_steps", type=int, default=30)

    # agent config
    parser.add_argument("--agent_type", type=str, default="prompt")
    parser.add_argument(
        "--instruction_path",
        type=str,
        default="agents/prompts/state_action_agent.json",
    )
    parser.add_argument(
        "--parsing_failure_th",
        help="When concesecutive parsing failure exceeds this threshold, the agent will stop",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--repeating_action_failure_th",
        help="When concesecutive repeating action exceeds this threshold, the agent will stop",
        type=int,
        default=3,
    )

    # lm config
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0613")
    parser.add_argument("--mode", type=str, default="chat")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--context_length", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=384)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument(
        "--max_retry",
        type=int,
        help="max retry times to perform generations when parsing fails",
        default=1,
    )
    parser.add_argument(
        "--max_obs_length",
        type=int,
        help="when not zero, will truncate the observation to this length before feeding to the model",
        default=1920,
    )
    parser.add_argument(
        "--model_endpoint",
        help="huggingface model endpoint",
        type=str,
        default="",
    )

    # example config
    parser.add_argument("--test_start_idx", type=int, default=0)
    parser.add_argument("--test_end_idx", type=int, default=1000)

    # logging related
    parser.add_argument("--result_dir", type=str, default="")

    parser.add_argument("--tools_return_true", action="store_false")
    parser.add_argument("--lc_type", type=str, default="default",
                        help="set to 'autogpt' for autogpt")
    parser.add_argument("--send_token_limit", type=int, default=4097, help="autogpt param")
    parser.add_argument("--base_plus_mem_tokens", type=int, default=2500, help="autogpt param for base prompt + memory")
    args = parser.parse_args()

    # check the whether the action space is compatible with the observation space
    if (
        args.action_set_tag == "id_accessibility_tree"
        and args.observation_type != "accessibility_tree"
    ):
        raise ValueError(
            f"Action type {args.action_set_tag} is incompatible with the observation type {args.observation_type}"
        )

    return args


def early_stop(
    trajectory: Trajectory, max_steps: int, thresholds: dict[str, int]
) -> tuple[bool, str]:
    """Check whether need to early stop"""

    # reach the max step
    num_steps = (len(trajectory) - 1) / 2
    if num_steps >= max_steps:
        return True, f"Reach max steps {max_steps}"

    last_k_actions: list[Action]
    action_seq: list[Action]

    # Case: parsing failure for k times
    k = thresholds["parsing_failure"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    if len(last_k_actions) >= k:
        if all(
            [
                action["action_type"] == ActionTypes.NONE
                for action in last_k_actions
            ]
        ):
            return True, f"Failed to parse actions for {k} times"

    # Case: same action for k times
    k = thresholds["repeating_action"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    action_seq = trajectory[1::2]  # type: ignore[assignment]

    if len(action_seq) == 0:
        return False, ""

    last_action: Action = action_seq[-1]

    if last_action["action_type"] != ActionTypes.TYPE:
        if len(last_k_actions) >= k:
            if all(
                [
                    is_equivalent(action, last_action)
                    for action in last_k_actions
                ]
            ):
                return True, f"Same action for {k} times"

    else:
        # check the action sequence
        if (
            sum([is_equivalent(action, last_action) for action in action_seq])
            >= k
        ):
            return True, f"Same typing action for {k} times"

    return False, ""


def test(
    args: argparse.Namespace,
    agent: Agent | PromptAgent | TeacherForcingAgent | LCAgent,
    config_file_list: list[str],
) -> None:
    scores = []
    max_steps = args.max_steps

    # my mods
    RESULT_FILE_PATH = f"{args.result_dir}/logged_results.csv"
    if os.path.exists(RESULT_FILE_PATH):
        print('results file exists, resuming from there')
        df = pd.read_csv(RESULT_FILE_PATH, index_col=0)
        df2 = df[~df.score.isnull()]
        for idx, score in zip(df2.index, df2.score):
            # print(idx, score)
            RESULT_TXT_PATH = f"{args.result_dir}/text_results/{int(idx)}_{int(score)}.txt"
            if not os.path.exists(RESULT_TXT_PATH):
                print(f'make result txt file {RESULT_TXT_PATH}')
                f = open(RESULT_TXT_PATH, "x")
                f.close()
    else:
        print('create new results file')
        df = pd.DataFrame(index=range(812), columns=['score'])


    early_stop_thresholds = {
        "parsing_failure": args.parsing_failure_th,
        "repeating_action": args.repeating_action_failure_th,
    }

    for config_file in config_file_list:
        try:
            my_globals.render_helper = RenderHelper(
                config_file, args.result_dir, args.action_set_tag
            )

            # get intent
            with open(config_file) as f:
                _c = json.load(f)
                intent = _c["intent"]
                task_id = _c["task_id"]
                # automatically login
                if _c["storage_state"]:
                    cookie_file_name = os.path.basename(_c["storage_state"])
                    comb = get_site_comb_from_filepath(cookie_file_name)
                    temp_dir = tempfile.mkdtemp()
                    # subprocess to renew the cookie
                    subprocess.run(
                        [
                            "python",
                            "browser_env/auto_login.py",
                            "--auth_folder",
                            temp_dir,
                            "--site_list",
                            *comb,
                        ]
                    )
                    _c["storage_state"] = f"{temp_dir}/{cookie_file_name}"
                    assert os.path.exists(_c["storage_state"])
                    # update the config file
                    config_file = f"{temp_dir}/{os.path.basename(config_file)}"
                    with open(config_file, "w") as f:
                        json.dump(_c, f)

            logger.info(f"[Config file]: {config_file}")
            logger.info(f"[Intent]: {intent}")

            agent.reset(config_file)
            my_globals.trajectory = []
            print("env reset\n")
            obs, info = my_globals.env.reset(options={"config_file": config_file})
            my_globals.state_info = {"observation": obs, "info": info}
            my_globals.trajectory.append(my_globals.state_info)

            my_globals.meta_data = {"action_history": ["None"]}
            # while True:
            for k in range(max_steps):
                print('check early stop')
                early_stop_flag, stop_info = early_stop(
                    my_globals.trajectory, max_steps, early_stop_thresholds
                )

                if early_stop_flag:
                    print('early stop')
                    action = create_stop_action(f"Early stop: {stop_info}")

                    my_globals.trajectory.append(action)
                    action_str = get_action_description(
                        action,
                        my_globals.state_info["info"]["observation_metadata"],
                        action_set_tag=args.action_set_tag,
                        prompt_constructor=None,
                    )
                    print(f'action str: {action_str}')
                    my_globals.render_helper.render(
                        action, my_globals.state_info, my_globals.meta_data, args.render_screenshot
                    )
                    my_globals.meta_data["action_history"].append(action_str)
                else:
                #     try:
                #         action = agent.next_action(
                #             trajectory, intent, meta_data=meta_data
                #         )
                #     except ValueError as e:
                #         # get the error message
                #         action = create_stop_action(f"ERROR: {str(e)}")

                    print("run agent \n")
                    hasError = False
                    try:
                        response = agent.run(my_globals.trajectory, intent, my_globals.meta_data)
                    except ValidationError as e:
                        response = {'output': f"Validation Error in args: {str(e)}"}
                        hasError = True
                    except Exception as e:
                        response = {'output': f"Error: {str(e)}, {type(e).__name__}"}
                        hasError = True
                        if args.lc_type == "autogpt":
                            raise e
                    if hasError or response['output'].startswith(my_globals.tool_error_start) or response['output'].startswith(my_globals.parse_error_start):
                        print("error \n")
                        my_globals.meta_data["action_history"].append(response['output'])

                    if isinstance(agent, LCAgent):
                        action = {"action_type": ActionTypes.NONE}
                        for j in range(len(my_globals.trajectory)-1, -1, -1):
                            if "action_type" in my_globals.trajectory[j]:
                                action = my_globals.trajectory[j]
                                break

                if action["action_type"] == ActionTypes.STOP:
                    break

                if args.lc_type == "autogpt":
                    action = create_stop_action(f"max steps in autogpt")
                    my_globals.trajectory.append(action)
                    break

            evaluator = evaluator_router(config_file)
            score = evaluator(
                trajectory=my_globals.trajectory,
                config_file=config_file,
                page=my_globals.env.page,
                client=my_globals.env.get_page_client(my_globals.env.page),
            )

            scores.append(score)

            df.score[task_id] = score
            for x in glob.glob(f"{args.result_dir}/text_results/*_*.txt"):
                new_idx, new_score = Path(x).stem.split('_')
                if df.score.isnull()[int(new_idx)]:
                    print('update new score frm parallel process')
                    df.score[int(new_idx)] = int(new_score)
            print(f'df stats: len: {len(df.index)} avg: {df.score.mean()} number of nan: {df.score.isnull().sum()}')
            df.to_csv(RESULT_FILE_PATH)
            f = open(f"{args.result_dir}/text_results/{int(task_id)}_{int(score)}.txt", "x")
            f.close()

            if 'full_msg_log' in response:
                with open(Path(args.result_dir) / f"msg_{task_id}.txt", "w") as file:
                    for line in response['full_msg_log']:
                        file.write(str(line) + "\n")

            if score == 1:
                logger.info(f"[Result] (PASS) {config_file}")
            else:
                logger.info(f"[Result] (FAIL) {config_file}")

            if args.save_trace_enabled:
                my_globals.env.save_trace(
                    Path(args.result_dir) / "traces" / f"{task_id}.zip"
                )

        except openai.error.OpenAIError as e:
            logger.info(f"[OpenAI Error] {repr(e)}")
        except Exception as e:
            logger.info(f"[Unhandled Error] {repr(e)}]")
            import traceback

            # write to error file
            with open(Path(args.result_dir) / f"error_{LOG_SUFFIX}.txt", "a") as f:
                f.write(f"[Config file]: {config_file}\n")
                f.write(f"[Unhandled Error] {repr(e)}\n")
                f.write(traceback.format_exc())  # write stack trace to file

        my_globals.render_helper.close()

    my_globals.env.close()
    logger.info(f"Average score: {sum(scores) / len(scores)}")


def prepare(args: argparse.Namespace) -> None:
    # convert prompt python files to json
    from agent.prompts import to_json

    to_json.run()

    # prepare result dir
    result_dir = args.result_dir
    if not result_dir:
        result_dir = (
            f"cache/results_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
        )
    if not Path(result_dir).exists():
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        args.result_dir = result_dir
        logger.info(f"Create result dir: {result_dir}")

    if not (Path(result_dir) / "traces").exists():
        (Path(result_dir) / "traces").mkdir(parents=True)

    if not (Path(result_dir) / "text_results").exists():
        (Path(result_dir) / "text_results").mkdir(parents=True)

    # log the log file
    with open(os.path.join(result_dir, "log_files.txt"), "a+") as f:
        f.write(f"{LOG_FILE_NAME}\n")


def get_unfinished(config_files: list[str], result_dir: str) -> list[str]:
    result_files = glob.glob(f"{result_dir}/*.html")
    task_ids = [
        os.path.basename(f).split(".")[0].split("_")[1] for f in result_files
    ]
    unfinished_configs = []
    for config_file in config_files:
        task_id = os.path.basename(config_file).split(".")[0]
        if task_id not in task_ids:
            unfinished_configs.append(config_file)
    return unfinished_configs


def dump_config(args: argparse.Namespace) -> None:
    config_file = Path(args.result_dir) / "config.json"
    if not config_file.exists():
        with open(config_file, "w") as f:
            json.dump(vars(args), f, indent=4)
            logger.info(f"Dump config to {config_file}")


if __name__ == "__main__":
    args = config()
    args.sleep_after_execution = 2.0
    prepare(args)

    test_file_list = []
    st_idx = args.test_start_idx
    ed_idx = args.test_end_idx
    for i in range(st_idx, ed_idx):
        test_file_list.append(f"config_files/{i}.json")
    if "debug" not in args.result_dir:
        test_file_list = get_unfinished(test_file_list, args.result_dir)

    if len(test_file_list) == 0:
        logger.info("No task left to run")
    else:
        print(f"Total {len(test_file_list)} tasks left")
        args.render_screenshot = True
        args.save_trace_enabled = True

        args.current_viewport_only = True
        dump_config(args)

        my_globals.init(args)

        agent = construct_agent(args)
        test(args, agent, test_file_list)
