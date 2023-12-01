from binhex import openrsrc
from curses import init_pair
import os
from unittest import result
import openai
import json
import argparse
import Levenshtein
# from nltk.translate.bleu_score import sentence_bleu
import sacrebleu
from time import sleep
import random

openai.api_key = os.getenv("OPENAI_API_KEY")
example ={
        "puzzle": ["A man walks into a bar and asks the bartender for a drink of water.",
                    "The bartender pulls out a qun, points it at the man and cocks it. ",
                    "The man pauses before saying \"Thank you\"and leaving.",
                    "What happened?"
        ],
        "final_answer": ["The man had the hiccups."
                  "The bartender realized this and chose instead to cure the hiccups by frightening the man with the gun."
        ],
        "question_list":["Could the bartender hear him?",
                       "Did the man ask for water in an offensive way?"],
        "answer_list":["Yes.","No."],
        "hint":"As we said, there is nothing wrong with either the building or the elevator. There is, however, some feature of the man that causes him to take the stairs from the seventh floor."
    }

def waitforGPT():
    sleep(30)

def load_data(data_path):
    with open(data_path, 'r') as f:
        data_set = json.load(f)
    return data_set

def extract_input_item(data):
    puzzle = "".join(data["puzzle"])
    truth = "".join(data["final_answer"])
    fol_q = list(data["question_list"])
    fol_a = list(data["answer_list"])
    return puzzle, truth, fol_q, fol_a

def get_response(prompt):
    output_ai = ""
    while output_ai == "":
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=prompt,
            temperature=0.7,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["\n\n"]
            )
        output_ai = response.get('choices')[0]['text'].split("\n")[0]
        if output_ai == "":
            waitforGPT()
    return output_ai


def solution_generate_input(data_item, shuffle):
    task_description = "I am an intelligent bot that can play situation puzzles with user. A puzzle is given first, and the user begin to ask \"yes/no\" question to ensure details, then I will give the question a\"yes/no/irrelevent\" answer. Finally user try to give solution for the puzzle."
    puzzle, truth, fol_q, fol_a = extract_input_item(data_item)
    if shuffle == "True":
        randomnum = random.randint(0,5)
        random.seed(randomnum)
        random.shuffle(fol_q)
        random.seed(randomnum)
        random.shuffle(fol_a)
    example_prefix = "puzzle:" + "".join(example["puzzle"]) + "\n"
    example_prefix += "follow_up_Q:"+example["question_list"][0]+"\n"+"follow_up_A:"+example["answer_list"][0] + "\n"
    example_prefix += "solution:" + "".join(example["final_answer"]) + "\n"
    input_prefix = "puzzle:" + puzzle + "\n"  
    if len(fol_q) and len(fol_a):
        for fq,fa in zip(fol_q, fol_a):
            input_prefix += "follow_up_Q:"+ fq +"\n"
            input_prefix += "follow_up_A:"+ fa+"\n"
        # input_prefix += "\n"
    input_prefix += "solution:"
    final_prefix = task_description + '\n' + example_prefix + input_prefix
    # print("---------- solution generation ----------")
    # print(final_prefix)
    return final_prefix

if __name__=="__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset_path', type=str, default="../data/merge_data.json")
    parse.add_argument('--sample_num', type=int, default=3)
    parse.add_argument('--chance_num', type=int, default=3)
    parse.add_argument('--threshold', type=int, default=0.85)
    parse.add_argument('--output_file', type=str, default="./output-50.txt")
    parse.add_argument('--output_dataset', type=str, default="./new_dataset.json")
    # parse.add_argument('--with_hint', action='store_true')
    args = parse.parse_args()
    dataset = load_data(args.output_dataset)
    for ditem in dataset[:args.sample_num]:
        print("PUZZLE:",ditem["puzzle"])
        # generate solution
        solution_generate_prompt = solution_generate_input(ditem, shuffle=False)
        # print(solution_generate_prompt)
        generated_solution = get_response(solution_generate_prompt)
        print("*",generated_solution)
        waitforGPT()
        # shuffle
        solution_generate_prompt_shuffle = solution_generate_input(ditem, shuffle=True)
        # print(solution_generate_prompt_shuffle)
        generated_solution_shuffle = get_response(solution_generate_prompt_shuffle)
        print("*shuffle*",generated_solution_shuffle)
        # calculate the current solution's accuracy：Jaro-Winkler
        similarity_score = Levenshtein.jaro_winkler("".join(ditem["final_answer"]),generated_solution)
        similarity_score_shuffle = Levenshtein.jaro_winkler("".join(ditem["final_answer"]),generated_solution_shuffle)
        print("Edit score: ", similarity_score,"\n", similarity_score_shuffle)
        similarity_score = sacrebleu.sentence_bleu("".join(ditem["final_answer"]),[generated_solution]).score
        similarity_score_shuffle = sacrebleu.sentence_bleu("".join(ditem["final_answer"]),[generated_solution_shuffle]).score
        print("BLEU score: ", similarity_score,"\n", similarity_score_shuffle)
        # calculate the current solution's accuracy：Bleu
