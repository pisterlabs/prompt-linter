"""
This module implements a standard experiment runner for all four-choice multiple choice question (MCQ) exams like
the MBE and MPRE.
"""

# imports
import datetime
import json
import logging
from pathlib import Path
from typing import Callable

# packages
import pandas
import tqdm

# project
from engines.base_engine import BaseEngine
from engines.engine_openai import OpenAIEngine
from exams.experiment_data import get_next_session_path

EXAM_PATH = Path(__file__).parent.parent.parent / "exams"

def load_sgml_essay_questions(
        exam_type: str, exam_id: str
) -> list[dict]:
    """Load a list of questions that look like this:
    <S>Section Name
    <Q>Question Prompt
    ...
    <S>Section Name
    <Q>Question Prompt
    ...

    Return a list of dictionaries like:
    [{"question_number": 1, "question_category": "Section Name", "question_prompt": "Question Prompt"}, ...]
    """
    # build path
    sgml_file = (
            EXAM_PATH
            / exam_type
            / exam_id
            / "questions.sgml"
    )

    # initialize the question list
    question_list = []

    # initialize the current question
    current_question = {}

    # iterate over the lines in the file
    current_section = None
    for line in sgml_file.read_text().splitlines():
        # strip the line
        line = line.strip()

        # skip empty lines
        if len(line) == 0:
            continue

        # check for the section tag
        if line.startswith("<S>"):
            # set the current section
            current_section = line[3:]

        # check for the question tag
        elif line.startswith("<Q>"):
            # set the current question
            current_question = {
                "question_number": len(question_list) + 1,
                "question_category": current_section,
                "question_prompt": line[3:],
            }

            # add the current question to the list
            question_list.append(current_question)
        else:
            # add the line to the current question
            current_question["question_prompt"] += f"\n{line}"

    # return the question list
    return question_list


def run_essay_exam(
    exam_type: str,
    exam_id: str,
    experiment_id: int,
    prompt_method: Callable,
    engine: OpenAIEngine | BaseEngine,
    num_samples: int = 1,
) -> list[Path]:
    """Run an exam session with the given parameters.

    N.B.: The engine is passed in as a parameter and no parameters are modified/swept within this method.
    Parameter setting/sweeping is responsibility of the caller.

    Args:
        exam_type (str): The exam type, e.g., mbe.
        exam_id (str): The exam id, e.g., ncbe-pdf-001
        experiment_id (int): The experiment id, e.g., 1.
        prompt_method (Callable): The prompt method to use.
        engine (BaseEngine): The engine to use.
        num_samples (int, optional): The number of samples to generate per question. Defaults to 1.

    Returns:
        list[Path]: A list of paths to the session files.
    """

    # load the questions from the exam
    question_data = load_sgml_essay_questions(exam_type, exam_id)

    # session path list
    session_path_list = []

    # get N samples
    for _ in range(num_samples):
        # get the next session path
        session_path = get_next_session_path(exam_type, exam_id, experiment_id)

        # create the session file path
        session_file_path = session_path / "session.json"

        # setup session data
        session_data = {
            "exam_type": exam_type,
            "exam_id": exam_id,
            "experiment_id": experiment_id,
            "prompt_method": prompt_method.__name__,
            "engine": engine.get_name(),
            "parameters": engine.get_current_parameters(),
            "session_start": datetime.datetime.now().isoformat(),
            "session_end": None,
            "question_data": [],
        }

        # iterate over the questions
        question_prog_bar = tqdm.tqdm(
            question_data, total=len(question_data)
        )
        for row_id, question_row in enumerate(question_prog_bar):
            # update the progress bar
            question_prog_bar.set_description(
                f"{exam_type}:{exam_id}:{experiment_id}:Q {row_id+1}/{len(question_data)}"
            )

            # initialize question record data
            question_row_data = {
                "question_category": question_row["question_category"],
                "question_number": question_row["question_number"],
                "question_prompt": question_row["question_prompt"],
                "model_prompt": None,
                "model_response": None,
            }

            # check whether we're in a chat or text completion API
            is_chat_model = False

            try:
                # get the question prompt
                prompt_values = prompt_method(question_row)
                if isinstance(prompt_values, str):
                    question_row_data["model_prompt"] = prompt_values
                    is_chat_model = False
                elif isinstance(prompt_values, tuple):
                    question_row_data["model_prompt"] = "<S>" + prompt_values[0] + "<U>" + prompt_values[1]

                    # check the token count against known model limits
                if engine.get_name() in ("openai:text-ada-001",
                                         "openai:text-babbage-001",
                                         "openai:text-curie-001",
                                         "openai:text-davinci-001",):
                    token_limit = 2049
                elif engine.get_name() in ("openai:text-davinci-003",
                                           "openai:text-davinci-002",
                                           "openai:gpt-3.5-turbo",
                                            # hidden-chat-gpt-model-name
                                           ):
                    token_limit = 4096
                else:
                    raise ValueError(f"Unknown engine name: {engine.get_name()}")

                # check the token count
                prompt_token_count = len(question_row_data["model_prompt"].split())
                token_request_count = int(token_limit - 1.5 * prompt_token_count)

                # get the model response
                if is_chat_model:
                    question_row_data["model_response"] = engine.get_completions(
                        system_prompt=prompt_values[0],
                        prompt=prompt_values[1],
                        parameters={"max_tokens": token_request_count},
                    )
                else:
                    question_row_data["model_response"] = engine.get_completions(
                        prompt=prompt_values,
                        parameters={"max_tokens": token_request_count},
                    )

            except Exception as e:
                logging.error(f"Error in question row_id={row_id}: {e}")
            finally:
                # add the question data
                session_data["question_data"].append(question_row_data)

                # save the session data each question
                with open(session_file_path, "w", encoding="utf-8") as f:
                    json.dump(session_data, f, indent=4)

        # update the session end time
        session_data["session_end"] = datetime.datetime.now().isoformat()

        # save the session data
        with open(session_file_path, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=4)
