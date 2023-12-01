import tqdm
import json
import openai
import logging
import argparse

from sklearn.metrics import accuracy_score

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAPPING = {"option1": "(1)", "option2": "(2)"}


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--openai_api_key",
        default=None,
        type=str,
        required=True,
        help="API key to use GPT-3.",
    )
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        required=True,
        help="The training set json file from which examples will be used for few-shot.",
    )
    parser.add_argument(
        "--eval_file",
        default=None,
        type=str,
        required=True,
        help="The dev/test set json file from which examples will be used for evaluation.",
    )
    parser.add_argument(
        "--out_prediction_file",
        default=None,
        type=str,
        required=True,
        help="Where to save the predictions.",
    )

    # Optional parameters
    parser.add_argument(
        "--num_train_examples",
        default=100,
        type=int,
        required=False,
        help="How many training examples. Note that the input will be truncated to 2048 tokens.",
    )
    args = parser.parse_args()

    openai.api_key = args.openai_api_key

    # Get the training examples - first num_train_examples examples from the training set
    train_examples = [json.loads(ex) for ex in open(args.train_file)][:args.num_train_examples]

    # Get the test examples
    test_examples = [json.loads(ex) for ex in open(args.eval_file)]

    predictions = solve(openai, train_examples, test_examples)
    golds = [MAPPING[ex["correctanswer"]] for ex in test_examples]

    # Save the predictions
    with open(args.out_prediction_file, "w") as f_out:
        for ex, pred, gold in zip(test_examples, predictions, golds):
            ex = {"input": ex["narrative"], "gold": gold, "predictions": [pred]}
            f_out.write(json.dumps(ex) + "\n")

    # Compute accuracy
    accuracy = accuracy_score(golds, predictions)
    logger.info(f"Accuracy: {accuracy*100.0:.1f}")


def create_prompt(ex, include_answer=False):
    """
    Creates a prompt for GPT-3
    """
    narrative = ex["narrative"].replace('<b>', '').replace('</b>', '')
    q = f"Q: {narrative}\n(1) {ex['option1']}\n(2) {ex['option2']}\n"
    a = f"A: {MAPPING[ex['correctanswer']]}\n###\n" if include_answer else "A:"
    return q + a


def solve(openai, train_examples, test_examples):
    """
    Train GPT-3 with few-shot learning
    :param openai: the OpenAI API object
    :param train_examples: list of dicts {"narrative", "plausible"}
    :param test_examples: list of dicts {"narrative", "plausible"}
    :return: the predictions
    """
    # Sort the training examples by the narrative length so we can include as many as possible
    sorted_train_examples = sorted(train_examples, key=lambda ex: len(ex["narrative"]))

    # Find the maximum length of test example so we can include it complete
    # (leave buffer because some words might be split to multiple tokens)
    max_len = max([len(ex["narrative"].split()) for ex in test_examples])
    train_prompt_len = int((2048 - max_len) * 0.9)

    prompt = ""

    for ex in sorted_train_examples:
        curr_prompt = create_prompt(ex, include_answer=True)

        if len(prompt) + len(curr_prompt) > train_prompt_len:
            break

        prompt += curr_prompt

    if len(prompt) > 2048:
        logger.warning("Question is incomplete")
        prompt = prompt[-2048:]

    logger.info(prompt)
    logger.info(f"Number of examples in practice: {len(prompt.split('###'))}")

    predictions = [
        openai.Completion.create(
            engine="davinci",
            prompt=f"{prompt}{create_prompt(ex, include_answer=False)}",
            temperature=0.7,
            max_tokens=3,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n"]
        )['choices'][0]['text'].strip()
        for ex in tqdm.tqdm(test_examples)
    ]

    return predictions


if __name__ == "__main__":
    main()
