import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from langchain.chains import LLMChain
from langchain.llms.base import LLM
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from pandera.errors import SchemaError

from lib.ai_eval_spreadsheet.schemas import (
    GenAiModel,
    GenAiModelConfig,
    PromptVariation,
    Question,
    QuestionOption,
)
from lib.ai_eval_spreadsheet.wrapper import (
    AiEvalData,
    get_ai_eval_spreadsheet,
    read_ai_eval_data,
)
from lib.app_singleton import AppSingleton
from lib.authorized_clients import get_service_account_authorized_clients
from lib.config import read_config
from lib.hash.fnv64hash import hash_dn
from lib.llms.utils import (
    get_alibaba_model,
    get_dummy_model,
    get_google_palm_model,
    get_huggingface_model,
    get_iflytek_model,
    get_openai_model,
)

logger = AppSingleton().get_logger()


# defining type alias to make function types more easier to write
QuestionAndOptions = Tuple[Question, List[QuestionOption]]
ModelAndConfig = Tuple[GenAiModel, GenAiModelConfig]


def read_ai_eval_spreadsheet() -> AiEvalData:
    config = read_config()
    authorized_clients = get_service_account_authorized_clients()

    ai_eval_spreadsheet_id = config["AI_EVAL_DEV_SPREADSHEET_ID"]
    ai_eval_spreadsheet = get_ai_eval_spreadsheet(
        authorized_clients, ai_eval_spreadsheet_id
    )
    try:
        return read_ai_eval_data(ai_eval_spreadsheet)
    except SchemaError as err:
        logger.error("DataFrame validation failed. Errors:", err.check)
        logger.error("Schema:")
        logger.error(err.schema)
        logger.error("Failure cases:")
        logger.error(err.failure_cases)  # dataframe of schema errors
        logger.error("Original data:")
        logger.error(err.data)  # invalid dataframe
        raise Exception("Data validation. Please fix and retry")


def filter_included_rows(df: pd.DataFrame) -> pd.DataFrame:
    """filter rows that are marked TRUE in `include_in_next_evaluation` column.

    the df will return as is if `include_in_next_evaluation` column not found.
    """
    col_to_filter = "include_in_next_evaluation"
    if col_to_filter not in df.columns:
        logger.warning("include_in_next_evaluation not found")
        return df
    return df[df[col_to_filter]]


def class_objects_from_df(df: pd.DataFrame, cls: type) -> list:
    # FIXME: how to write correct type annotation for this?
    return [cls(**rec) for rec in df.to_dict(orient="records")]


def get_questions(
    sheet: AiEvalData, include_all: bool = False, language: Optional[str] = None
) -> List[QuestionAndOptions]:
    if include_all:
        questions = sheet.questions.data.df
    else:
        questions = filter_included_rows(sheet.questions.data.df)

    if language is not None:
        questions = questions.loc[questions["language"] == language]

    options = sheet.question_options.data.df
    qs = class_objects_from_df(questions, Question)

    res = []
    for q in qs:
        qid = q.question_id
        lang = q.language
        qopts = [
            QuestionOption(**rec)
            for rec in options.loc[
                (options["question_id"] == qid) & (options["language"] == lang)
            ].to_dict(orient="records")
        ]
        res.append((q, qopts))

    return res


def get_model(model_id, vendor, model_conf):
    if vendor == "OpenAI":
        return get_openai_model(model_id, **model_conf)
    if vendor == "Google":
        return get_google_palm_model(model_id, **model_conf)
    elif vendor == "Dummy":
        return get_dummy_model(model_id, **model_conf)
    elif vendor == "HuggingFace":
        return get_huggingface_model(model_id, **model_conf)
    elif vendor == "iFlyTek":
        return get_iflytek_model(**model_conf)
    elif vendor == "Alibaba":
        return get_alibaba_model(model_id, **model_conf)
    else:
        raise NotImplementedError(f"{model_id} from {vendor} is not supported yet.")


def option_text(opt: QuestionOption, letter_and_text: bool = True) -> str:
    if letter_and_text:
        return f"{opt.letter}. {opt.question_option}"
    else:
        return opt.letter


def simple_evaluation(question: QuestionAndOptions, answer: str) -> str:
    correctness_map = {1: "correct", 2: "wrong", 3: "very wrong"}
    # sometimes the model will return 'A.' instead of 'A'
    # and sometimes model will return 'A. 20%.' instead of 'A. 20%'
    answer = answer.strip()
    if answer[-1] == ".":
        answer = answer[:-1]

    for opt in question[1]:
        if answer == opt.letter or answer == option_text(opt, letter_and_text=True):
            return correctness_map[opt.correctness_of_answer_option]
        elif option_text(opt, letter_and_text=True) in answer:
            logger.debug("not exact match but the answer includes the option")
            logger.debug(answer)
            return correctness_map[opt.correctness_of_answer_option]
    return "failed"


def create_question_data_for_test(
    question_tmpl: str, question: QuestionAndOptions
) -> Dict[str, str]:
    q, options = question
    question_dict = {"question": q.published_version_of_question}
    for opt in options:
        if opt.letter == "A":
            question_dict["option_a"] = opt.question_option
        elif opt.letter == "B":
            question_dict["option_b"] = opt.question_option
        elif opt.letter == "C":
            question_dict["option_c"] = opt.question_option
    return {"question": question_tmpl.format(**question_dict)}


def create_question_dataset_for_test(
    question_tmpl: str,
    question_list: List[QuestionAndOptions],
) -> List[Dict[str, str]]:
    return [create_question_data_for_test(question_tmpl, q) for q in question_list]


def create_question_data_for_eval(question: QuestionAndOptions) -> Dict[str, str]:
    q, options = question
    question_dict = {"question": q.published_version_of_question}
    for opt in options:
        if opt.letter == "A":
            question_dict["option_a"] = opt.question_option
            question_dict["option_a_correctness"] = opt.correctness_of_answer_option
        elif opt.letter == "B":
            question_dict["option_b"] = opt.question_option
            question_dict["option_b_correctness"] = opt.correctness_of_answer_option
        elif opt.letter == "C":
            question_dict["option_c"] = opt.question_option
            question_dict["option_c_correctness"] = opt.correctness_of_answer_option
    return question_dict


def create_question_dataset_for_eval(
    question_list: List[QuestionAndOptions],
) -> List[Dict[str, str]]:
    return [create_question_data_for_eval(q) for q in question_list]


def check_llm_eval_output(eval_output: str) -> str:
    eval_output = eval_output.strip().replace(".", "").lower()[0]
    if eval_output == "1":
        return "correct"
    elif eval_output == "2":
        return "wrong"
    elif eval_output == "3":
        return "very wrong"
    else:
        return "failed"


def get_prompt_variants(
    sheet: AiEvalData, include_all: bool = False
) -> List[PromptVariation]:
    if include_all:
        prompt_variations = sheet.prompt_variations.data.df
    else:
        prompt_variations = filter_included_rows(sheet.prompt_variations.data.df)
    res = class_objects_from_df(prompt_variations, PromptVariation)
    return res


def get_model_configs(sheet: AiEvalData) -> List[ModelAndConfig]:
    models_df = sheet.gen_ai_models.data.df
    model_configs_df = filter_included_rows(sheet.gen_ai_model_configs.data.df)

    model_configs = class_objects_from_df(model_configs_df, GenAiModelConfig)
    result = []
    for mc in model_configs:
        model_df = models_df.loc[models_df["model_id"] == mc.model_id]
        model = class_objects_from_df(model_df, GenAiModel)[0]
        result.append((model, mc))
    return result


def get_survey_hash(questions: List[QuestionAndOptions]) -> str:
    joined = ",".join([q[0].question_id for q in questions])
    return hash_dn(joined, "")


def load_model_parameters(s: str) -> Dict[str, Any]:
    if s == "nan":
        # NOTE: nan (float) value has converted to 'nan' (string)
        # by the reader. That's why I am checking with 'nan' here.
        return {}
    return json.loads(s)


def run_survey(
    survey: Tuple[str, List[QuestionAndOptions]],
    prompt_var: PromptVariation,
    model_conf: ModelAndConfig,
    eval_llm: LLM,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    model, conf = model_conf
    model_id = model.model_id
    model_parameters = load_model_parameters(conf.model_parameters)
    prompt_id = prompt_var.variation_id
    model_config_id = conf.model_config_id
    vendor = model.vendor
    llm = get_model(model_id, vendor, model_parameters)
    prompt_tmpl = PromptTemplate.from_template(prompt_var.question_prompt_template)
    question_tmpl = prompt_var.question_template

    if conf.memory:
        memory = ConversationBufferWindowMemory(
            k=conf.memory_size,
            human_prefix=prompt_var.question_prefix,
            ai_prefix=prompt_var.ai_prefix,
        )
        chain = LLMChain(llm=llm, prompt=prompt_tmpl, memory=memory, verbose=verbose)
    else:
        chain = LLMChain(llm=llm, prompt=prompt_tmpl, verbose=verbose)

    followup = prompt_var.follow_up_answer_correctness_evaluation_prompt_template

    results = []
    session_id = str(uuid.uuid4())
    session_time = datetime.isoformat(datetime.utcnow())
    # 1. get output from LLM.
    # 2. get grade.
    survey_id, questions = survey
    log_msg = [
        "Evaluating:",
        f"Model: {model_id}",
        f"parameters: {model_parameters}",
        f"memory: {conf.memory}",
        f"Survey ID: {survey_id}",
        f"prompt ID: {prompt_id}",
    ]
    logger.info("\n".join(log_msg))
    if followup == "nan":
        logger.debug("using simple string matching to correctness")
    else:
        logger.debug("using LLM method to check correctness")
    for i, question in enumerate(questions):
        res = {
            "session_id": session_id,
            "session_time": session_time,
            "survey_id": survey_id,
            "model_configuration_id": model_config_id,
            "prompt_variation_id": prompt_id,
            "question_id": question[0].question_id,
            "language": question[0].language,
            "question_number": i + 1,
        }
        question_data = create_question_data_for_test(question_tmpl, question)
        eval_data = create_question_data_for_eval(question)
        output = chain.run(question_data)
        res["output"] = output
        if followup == "nan":  # simple string matching
            # NOTE: nan (float) value has converted to 'nan' (string)
            # by the reader. That's why I am checking with 'nan' here.
            res["grade"] = simple_evaluation(question, output)
        else:  # use LLM to eval
            followup_tmpl = PromptTemplate.from_template(followup)
            eval_chain = LLMChain(llm=eval_llm, prompt=followup_tmpl, verbose=verbose)
            # combine the output and eval dataset
            eval_data["text"] = output
            grade_output = eval_chain.run(eval_data)
            logger.debug("eval llm output: " + grade_output)
            grade = simple_evaluation(question, grade_output)
            res["grade"] = grade
        results.append(res)
    return results


def run_survey_n_round(
    survey: Tuple[str, List[QuestionAndOptions]],
    prompt_var: PromptVariation,
    model_conf: ModelAndConfig,
    eval_llm: LLM,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    repeat_times = model_conf[1].repeat_times
    result = []
    for _ in range(repeat_times):
        result.extend(run_survey(survey, prompt_var, model_conf, eval_llm, verbose))
    return result
