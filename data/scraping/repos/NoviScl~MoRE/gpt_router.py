import os
import argparse
import numpy as np
from time import sleep
import json
import random
import openai
from tqdm import tqdm

test_dir = "uniqa_predictions_final/test/"
dev_dir = "uniqa_predictions_final/dev/"
datasets = [
    "nq",
    "triviaqa",
    "squad",
    "hotpotqa",
    "beerqa_3hop",
    "musique",
    "gsm8k",
    "svamp",
    "multiarith",
    "csqa",
    "csqa2",
    "qasc"
]

experts = ["factual", "multihop", "math", "commonsense"]
choices = ["A", "B", "C", "D"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apikey', type=str, help='api key; https://openai.com/api/')
    parser.add_argument('--engine', type=str, default='text-davinci-003', help='api engine; https://openai.com/api/')
    parser.add_argument('--maxlen', type=int, default=1, help='max len')
    args = parser.parse_args()
    openai.api_key = args.apikey

    prompt = ""
    prompt += "I am in a question-answering contest. There are four experts on my team: expert A specializes in factoid questions that focus on factual knowledge; expert B specializes in multihop questions that need multiple steps of reasoning; expert C specializes in mathematics questions that require logical reasoning and math calculation; expert D specializes in commonsense questions that focus on commonsense knowledge.\n"
    prompt += "Given a question, I will get answers from all four experts, your task is to choose the best answer from the four answers. Directly predict the index of the chosen expert (A, B, C, or D). If there are multiple correct answers, just choose one of them.\n\n"

    with open("sampled_demos.json", 'r') as f:
        sampled_examples = json.load(f)
    for dp in sampled_examples:
        prompt += "Question: " + dp["question"] + "\n"
        prompt += "A: " + dp["factual"].replace("\n", " ").strip() + "\n"
        prompt += "B: " + dp["multihop"].replace("\n", " ").strip() + "\n"
        prompt += "C: " + dp["math"].replace("\n", " ").strip() + "\n"
        prompt += "D: " + dp["commonsense"].replace("\n", " ").strip() + "\n"
        prompt += "Best Answer: " + dp["answer"].strip() + "\n"
        prompt += "\n"
    
    all_gpt_preds = {}
    for dataset in datasets:
        correct = 0
        total = 0
        all_dp = {"dataset": dataset, "data": []}
        all_data = {}
        for expert in experts:
            fname = os.path.join(test_dir, dataset + '_' + expert + ".json")
            with open(fname, 'r') as f:
                data = json.load(f)
            all_data[expert] = data
        
        for i in tqdm(range(len(all_data["factual"]))):
            total += 1
            EMs = []
            dp = {}
            cur_prompt = prompt
            cur_prompt += "Question: " + all_data["factual"][i]["question"].strip() + "\n"
            dp["question"] = all_data["factual"][i]["question"].strip()
            
            if all_data["factual"][i]["full_answer"]:
                cur_prompt += "A: " + all_data["factual"][i]["full_answer"].replace("\n", " ").strip() + "\n"
            else:
                cur_prompt += "A: " + all_data["factual"][i]["answer"].replace("\n", " ").strip() + "\n"
            EMs.append(all_data["factual"][i]["em"])
            dp["factual"] = all_data["factual"][i]["answer"].replace("\n", " ").strip()
            
            if all_data["multihop"][i]["full_answer"]:
                cur_prompt += "B: " + all_data["multihop"][i]["full_answer"].replace("\n", " ").strip() + "\n"
            else:
                cur_prompt += "B: " + all_data["multihop"][i]["answer"].replace("\n", " ").strip() + "\n"
            EMs.append(all_data["multihop"][i]["em"])
            dp["multihop"] = all_data["multihop"][i]["answer"].replace("\n", " ").strip()

            if all_data["math"][i]["full_answer"]:
                cur_prompt += "C: " + all_data["math"][i]["full_answer"].replace("\n", " ").strip() + "\n"
            else:
                cur_prompt += "C: " + all_data["math"][i]["answer"].replace("\n", " ").strip() + "\n"
            EMs.append(all_data["math"][i]["em"])
            dp["math"] = all_data["math"][i]["answer"].replace("\n", " ").strip()
            
            if all_data["commonsense"][i]["full_answer"]:
                cur_prompt += "D: " + all_data["commonsense"][i]["full_answer"].replace("\n", " ").strip() + "\n"
            else:
                cur_prompt += "D: " + all_data["commonsense"][i]["answer"].replace("\n", " ").strip() + "\n"
            EMs.append(all_data["commonsense"][i]["em"])
            dp["commonsense"] = all_data["commonsense"][i]["answer"].replace("\n", " ").strip()
            dp["EMs"] = EMs
            
            cur_prompt += "Best Answer: "

            # print (cur_prompt)
            # print ("\n\n\n")

            response = None
            while response is None:
                try:
                    response = openai.Completion.create(
                        engine=args.engine,
                        prompt=cur_prompt,
                        max_tokens=args.maxlen,
                        logprobs=5,
                        temperature=0.,
                        stream=False,
                        stop=["<|endoftext|>", "\n"]
                    )
                except:
                    sleep(10)
                    continue

            output = response['choices'][0]["text"].strip()
            try:
                raw_logprobs = response['choices'][0]["logprobs"]["top_logprobs"][0]
            except:
                raw_logprobs = []

            # print (output)
            # print (raw_logprobs)
            # print ("\n\n")

            if output not in choices:
                output = random.choice(choices)
            dp["choice"] = output
            dp["EM"] = EMs[choices.index(output)]
            correct += EMs[choices.index(output)]
        
            all_dp["data"].append(dp)

        accuracy = correct / total * 100 
        print ("Accuracy on {}: {} / {} = {}%".format(dataset, correct, total, accuracy))
        all_gpt_preds[dataset] = all_dp

    with open("gpt_preds.json", 'w') as f:
        json.dump(all_gpt_preds, f, indent=4)

if __name__ == '__main__':
    main()