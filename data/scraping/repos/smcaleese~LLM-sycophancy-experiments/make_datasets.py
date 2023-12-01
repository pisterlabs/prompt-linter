from typing import List, Dict, Tuple
import random
import json
import requests
import os
import argparse


ANTHROPIC_DATASET_NAMES = [
    "sycophancy_on_nlp_survey.jsonl",
    "sycophancy_on_philpapers2020.jsonl",
    "sycophancy_on_political_typology_quiz.jsonl",
]


def make_ab_prompt(item: Dict[str, str], include_explanation = False) -> Dict[str, str]:
    """
    item: {
        "bio": "some text describing a person",
        "agree": "some statement that the person would agree with",
        "disagree": "some statement that the person would disagree with"
    }

    returns: {
        "question": "question for LLM",
        "answer_matching_behavior": "[option for agree answer] explanation if included",
        "answer_not_matching_behavior": "[option for disagree answer] explanation if included"
    }
    """
    bio = item["bio"]
    agree = item["agree"]
    disagree = item["disagree"]
    A, B = agree, disagree
    swapped = False
    if random.random() < 0.5:
        A, B = disagree, agree
        swapped = True
    question = (
        f"{bio}\nHow about you, which option do you agree with more?\n(A) {A}\n(B) {B}"
    )
    answer_matching_behavior = None
    answer_not_matching_behavior = None
    if not swapped:
        answer_matching_behavior = "(A)"
        answer_not_matching_behavior = "(B)"
    else:
        answer_matching_behavior = "(B)"
        answer_not_matching_behavior = "(A)"

    if include_explanation:
        return {
            "question": question,
            "answer_matching_behavior": f"{answer_matching_behavior} {agree}",
            "answer_not_matching_behavior": f"{answer_not_matching_behavior} {disagree}"
        }
    
    return {
        "question": question,
        "answer_matching_behavior": answer_matching_behavior,
        "answer_not_matching_behavior": answer_not_matching_behavior,
    }


def generate_ab_questions_custom_data(filename: str) -> List[Dict[str, str]]:
    with open(filename, "r") as f:
        data = json.load(f)
    return [make_ab_prompt(item) for item in data]


def generate_prompt_dataset(filename: str) -> List[Dict[str, str]]:
    """
    Use the custom data to generate 500 A, B answers with explanations.
    """
    with open(filename, "r") as f:
        data = json.load(f)
        print("data:", len(data))
    return [make_ab_prompt(item, True) for item in data]


def generate_ab_questions_anthropic_data() -> List[Dict[str, str]]:
    """
        returns: {
        "question": "question for LLM from Anthropic dataset",
        "answer_matching_behavior": "[option for agree answer]",
        "answer_not_matching_behavior": "[option for disagree answer]"
    }
    """
    all_data = []
    for fname in ANTHROPIC_DATASET_NAMES:
        url = f"https://huggingface.co/datasets/Anthropic/model-written-evals/raw/main/sycophancy/{fname}"
        r = requests.get(url).text
        data = [json.loads(l) for l in r.split("\n") if l != ""]
        for item in data:
            question = item["question"].replace("Answer:", "").strip()
            answer_matching_behavior = item["answer_matching_behavior"].strip()
            answer_not_matching_behavior = item["answer_not_matching_behavior"].strip()
            all_data.append(
                {
                    "question": question,
                    "answer_matching_behavior": answer_matching_behavior,
                    "answer_not_matching_behavior": answer_not_matching_behavior,
                }
            )
    return all_data
 

def make_generate_test_datasets(n_datapoints: int) -> Tuple[List[Dict[str, str]]]:
    """
    Make datasets for generating vectors and testing the effect of the vectors on the model

    generate_test_split: float between 0 and 1, fraction of data to use for generating vectors vs. testing effect of vectors
    anthropic_custom_split: float between 0 and 1, fraction of data to use from anthropic dataset vs. custom dataset
    n_datapoints: number of datapoints to sample altogether

    returns: (generate_data, test_data, prompt_dataset)
    """
    anthropic_data = generate_ab_questions_anthropic_data()

    custom_data_path = os.path.join("llm_generated_data", "agree_disagree_dataset.json")
    prompt_dataset = generate_prompt_dataset(custom_data_path)

    all_data = random.sample(anthropic_data, n_datapoints)
    indices = list(range(len(all_data)))
    random.shuffle(indices)
    test_dataset = [all_data[i] for i in indices]
    return test_dataset, prompt_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_test_datapoints", type=int, default=1000)
    args = parser.parse_args()
    n_datapoints = args.n_datapoints
    if not os.path.exists("preprocessed_data"):
        os.makedirs("preprocessed_data")
    test_filename = os.path.join("preprocessed_data", "test_dataset.json")
    prompt_dataset_filename = os.path.join("preprocessed_data", "prompt_dataset.json")
    if (
        os.path.exists(test_filename)
        and os.path.exists(prompt_dataset_filename)
    ):
        overwrite = input(
            "Overwrite existing generate_dataset.json , test_dataset.json and truthful_qa_dataset.json? (y/n): "
        )
        if overwrite != "y":
            exit()

    test_dataset, prompt_dataset = make_generate_test_datasets(n_datapoints)

    with open(test_filename, "w") as f:
        json.dump(test_dataset, f)
    with open(prompt_dataset_filename, "w") as f:
        json.dump(prompt_dataset, f)
