from datasets import load_dataset
import pandas as pd
from tqdm.auto import tqdm
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import dotenv
import os
import time

dotenv.load_dotenv()


def evaluate_guess(question, guess, answer):
    anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    completion = anthropic.completions.create(
        model="claude-2.1",
        max_tokens_to_sample=1,
        prompt=f"Answer either Y or N, and nothing else.{HUMAN_PROMPT}Are these two answers essentially equivalent as an answer to this question: {question}? Answer A: {guess}, Answer B: {answer}{AI_PROMPT}",
    )
    return completion.completion.lower().strip().startswith("y")


class AnthropicModel:
    def __init__(self, model):
        self.anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model

    def generate_guess(self, system, question):
        completion = self.anthropic.completions.create(
            model=self.model,
            max_tokens_to_sample=300,
            prompt=f"{system}{HUMAN_PROMPT}{question}{AI_PROMPT}A:",
        )
        return completion.completion


dataset = load_dataset("MemGPT/MSC-Self-Instruct")

previous_dialogues = dataset["train"]["previous_dialogs"][0]

model = AnthropicModel("claude-2.1")


if os.path.exists("dialogue_data.csv"):
    dialogue_data = pd.read_csv("dialogue_data.csv")
    if "correct_answer" not in dialogue_data.columns:
        dialogue_data["correct_answer"] = [None] * len(dialogue_data)
else:
    dialogue_data = pd.DataFrame(
        columns=["system", "question", "answer", "correct", "time", "correct_answer"]
    )

length = len(list(dataset["train"]["previous_dialogs"]))
for row, dialogue in tqdm(
    enumerate(dataset["train"]["previous_dialogs"]), total=length
):
    full_dialogue = []
    for d in dialogue:
        conversation = d["dialog"]
        for i, turn in enumerate(conversation):
            speaker = "A" if i % 2 == 0 else "B"
            full_dialogue.append(f"{speaker}: \"{turn['text']}\"\n")
    system = f"Based on the following dialogue:\n{''.join(full_dialogue)}\n please answer what A would say to the provided question. Be as concise as possible, you don't need to mimick the conversational style above."
    question = (
        "What would A respond to this: " + dataset["train"]["self_instruct"][row]["B"]
    )
    correct_answer = dataset["train"]["self_instruct"][row]["A"]
    if not (
        (dialogue_data["system"] == system) & (dialogue_data["question"] == question)
    ).any():
        start_time = time.time()
        answer = model.generate_guess(system, question)
        end_time = time.time()
        elapsed_time = end_time - start_time
        correct = evaluate_guess(question, answer, correct_answer)
        new_row = pd.DataFrame(
            {
                "system": [system],
                "question": [question],
                "answer": [answer],
                "correct": [correct],
                "time": [elapsed_time],
                "correct_answer": [correct_answer],
            }
        )
        dialogue_data = pd.concat([dialogue_data, new_row], ignore_index=True)
        if row % 10 == 0:
            dialogue_data.to_csv("dialogue_data.csv", index=False)
    else:
        index = dialogue_data[
            (dialogue_data["system"] == system)
            & (dialogue_data["question"] == question)
        ].index[0]
        if pd.isnull(dialogue_data.loc[index, "correct_answer"]):
            dialogue_data.loc[index, "correct_answer"] = correct_answer
            dialogue_data.to_csv("dialogue_data.csv", index=False)
        if pd.isnull(dialogue_data.loc[index, "correct"]):
            answer = dialogue_data.loc[index, "answer"]
            correct_answer = dialogue_data.loc[index, "correct_answer"]
            correct = evaluate_guess(question, answer, correct_answer)
            dialogue_data.loc[index, "correct"] = correct
            dialogue_data.to_csv("dialogue_data.csv", index=False)
