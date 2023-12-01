import json
import argparse
from typing import List, Dict, Tuple
from topic_lines import topic_line_mapping
from anthropic import HUMAN_PROMPT, AI_PROMPT
import os

def template(question: str, choices: List[str], answers: List[str] = None, topic_line: str = "") -> str:
    choices_prompt = ""
    for i, choice in enumerate(choices):
        choices_prompt += f"({chr(i + ord('A'))}) {choice}\n"
    if answers:
        return f"{HUMAN_PROMPT}{topic_line}問題：{question}\n{choices_prompt}正確答案：{AI_PROMPT}({')('.join(answers)})"
    else:
        return f"{HUMAN_PROMPT}{topic_line}問題：{question}\n{choices_prompt}正確答案：{AI_PROMPT}"

def parse_problem(problem: Dict[str, str]) -> Tuple[str, List[str], List[str]]:
    question = problem["question"]
    question = question.replace("\n\n", "\n")
    correct_choices = problem["correct_choices"]
    incorrect_choices = problem["incorrect_choices"]
    choices: List[Tuple[str, bool]] = []
    for choice in correct_choices:
        choices.append((choice, True))
    for choice in incorrect_choices:
        choices.append((choice, False))
    choices.sort()

    answers: List[str] = []
    for i, choice in enumerate(choices):
        if choice[1]:
            answers.append(chr(i + ord("A")))
    choices = [choice[0] for choice in choices]
    return question, choices, answers

def format_problem(problem: Dict[str, str], 
                   topic_line: str ="以下題目為選擇題，答案可能為單個或多個。",
                   examples: List[Dict[str, str]] = None) -> Tuple[str, List[str]]:
    question, choices, answers = parse_problem(problem)
    # prompt = topic_line + "\n\n"
    prompt = ""
    if examples:
        for i, example in enumerate(examples):
            ex_question, ex_choices, ex_answers = parse_problem(example)
            if i == 0:
                ex_problem_prompt = template(ex_question, ex_choices, ex_answers, topic_line+"\n\n")
            else:
                ex_problem_prompt = template(ex_question, ex_choices, ex_answers)
            prompt += ex_problem_prompt
    problem_prompt = template(question, choices)
    prompt += problem_prompt
    
    return prompt, answers

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_jsonl", 
        type=str, 
        required=True,
        help="Path to the imput jsonl for all questions."
    )
    parser.add_argument(
        "--example_jsonl", 
        type=str, 
        default="", 
        required=False, 
        help="Path to the example jsonl for few-shot demonstrations."
    )
    parser.add_argument(
        "--prompt_jsonl", 
        type=str, 
        required=True, 
        help="Output jsonl that will be used to store the formated prompts."
    )

    args = parser.parse_args()

    input_file = os.path.basename(args.input_jsonl)
    if topic_line_mapping.get(input_file):
        topic_line = topic_line_mapping[input_file]
    else:
        topic_line = "以下題目為選擇題，答案可能為單個或多個。"

    if args.example_jsonl:
        examples = []
        example_ids = set()
        with open(args.example_jsonl, "r", encoding="UTF-8") as f:
            examples_jsonl = f.readlines()
            for line in examples_jsonl:
                examples_problem = json.loads(line)
                examples.append(examples_problem)
                example_ids.add(examples_problem["id"])
    else:
        examples = None

    output = []
    with open(args.input_jsonl, "r", encoding="UTF-8") as f:
        input_jsonl = f.readlines()
        for line in input_jsonl:
            problem = json.loads(line)
            if args.example_jsonl and problem["id"] in example_ids:
                continue
            else:
                prompt, answer = format_problem(problem, topic_line, examples)
                output.append({
                    "id": problem["id"],
                    "prompt": prompt,
                    "answer": answer
                })
    
    with open(args.prompt_jsonl, "w", encoding="UTF-8") as f:
        for prediction in output:
            line = json.dumps(prediction, ensure_ascii=False)
            f.write(line+"\n")

    
    