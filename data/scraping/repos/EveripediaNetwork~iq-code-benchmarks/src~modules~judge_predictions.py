from pandas import DataFrame
from predictors.gpt_4_turbo import gpt_4_turbo
from lib.prompts import judge_predictions_prompt
from tqdm import tqdm
from json import loads
from termcolor import cprint
from multiprocessing import Pool


def judge_predictions(df: DataFrame, llm_choices: list[str]) -> DataFrame:
    """
    Compare the predictions to the actual issues from the dataset
    and appends {name-of-llm}_correct columns to the DataFrame
    which contains list of issue indices that were correctly predicted
    """

    # Initialize columns in the DataFrame for each judgement type and llm_choice
    for llm_name in llm_choices:
        for judgement_type in ["false_negatives", "false_positives", "true_positives"]:
            column_name = f"{llm_name}_{judgement_type}"
            if column_name not in df.columns:
                df[column_name] = [None] * len(df)

    # prepare the arguments for multiprocessing
    tasks = []
    for llm in llm_choices:
        for loc, row in df.iterrows():
            predicted_issues = row[f"{llm}_prediction"]
            actual_issues = row["issues"]
            tasks.append((loc, llm, actual_issues, predicted_issues))

    # Use multiprocessing Pool to run tasks in parallel
    judgments = []
    with Pool(8) as pool:
        p_bar = tqdm(
            pool.imap_unordered(make_judgement, tasks),
            total=len(tasks),
            desc="🔎 Judging predictions",
        )
        for result in p_bar:
            judgments.append(result)

    # Update the DataFrame with the judgments
    for loc, llm_name, judgement in judgments:
        # here at each df loc, we insert false_negatives, false_positives, true_positives to their respective columns
        for key, value in judgement.items():
            df.at[loc, f"{llm_name}_{key}"] = value

    return df


def make_judgement(args) -> tuple[int, str, dict]:
    """
    Compare the predictions to the actual issues and return the indices of the
    issues that were correctly predicted. This uses gpt-4-turbo to compare the
    issues.
    """
    loc, llm_name, actual_issues, predicted_issues = args
    result = gpt_4_turbo(
        judge_predictions_prompt.format(
            key=format_key(actual_issues), prediction=predicted_issues
        )
    )

    try:
        # remove the prompt from the result (only keep the json which should start from first { and end with last })
        result = result[result.find("{") : result.rfind("}") + 1]

        # parse the json result
        result = loads(result)

        # return the indices of the issues that were correctly predicted
        # we are doing this to make sure the format is consistent or else we produce error.
        # see if we can do this better with output parsers from langchain
        parsed_result = {
            "false_negatives": result["false_negatives"],
            "false_positives": result["false_positives"],
            "true_positives": result["true_positives"],
        }

        return loc, llm_name, parsed_result
    except:
        cprint(f"🚨 Error parsing judgement: {result}", "white", "on_red")
        error_result = {
            "false_negatives": None,
            "false_positives": None,
            "true_positives": None,
        }
        return loc, llm_name, error_result


def format_key(issues: list[str]) -> str:
    """
    Format the issues list to be displayed in the prompt
    """
    key = []
    for i, issue in enumerate(issues):
        category = issue["category"]
        description = issue["description"]
        location = issue["location"]
        impact = issue["impact"]

        key.append(
            f"{i + 1}) category: {category}\ndescription: {description}\nlocation: {location}\nimpact: {impact}\n"
        )
    return "\n".join(key)
