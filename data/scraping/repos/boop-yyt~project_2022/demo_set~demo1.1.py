from binhex import openrsrc
from curses import init_pair
import os
from pickle import FALSE, TRUE
from time import sleep
from unittest import result
import openai
import json
import argparse
import Levenshtein

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

def example_truncation(example_prefix, input_prefix, TOKEN_MAXLEN):
    example_prefix_len, input_prefix_len = len(example_prefix), len(input_prefix)
    if example_prefix_len + input_prefix_len > TOKEN_MAXLEN:
        example_start = example_prefix_len - TOKEN_MAXLEN - input_prefix_len
    else: 
        example_start = 0
    return example_start

def question_generation_input(data_item, THINK_HINT, TOKEN_MAXLEN, example_set):
    # task_description = "I am an intelligent bot that can play situation puzzles with user. A puzzle is given first, and the user begin to ask a \"yes/no\" question to ensure details."
    task_description = "I can generate five questions based on the \"puzzle\" that start with \"Do/Am/Is/Are/Can/Could/Does/Did/Was/Were\" and can only be answered with \"yes\" or \"no\"."
    puzzle, truth, fol_q, fol_a = extract_input_item(data_item)
    example_prefix = "puzzle:" + "".join(example["puzzle"]) + "\n"
    example_prefix += "follow_up_Q:"+example["question_list"][0]+"\n"+"follow_up_A:"+example["answer_list"][0] + "\n"
    example_prefix += "follow_up_Q:"+example["question_list"][1] + "\n"
    input_prefix = "puzzle:" + puzzle + "\n"  
    if len(fol_q) and len(fol_a):
        for fq,fa in zip(fol_q, fol_a):
            input_prefix += "follow_up_Q:"+ fq
            input_prefix += "follow_up_A:"+ fa
        input_prefix += "\n"
    input_prefix += THINK_HINT + "follow_up_Q:"
    example_start = example_truncation(example_prefix=example_prefix,input_prefix=input_prefix, TOKEN_MAXLEN=TOKEN_MAXLEN)
    if example_set==TRUE:
        final_prefix = task_description + '\n' + example_prefix[example_start:] + input_prefix
    else:
        final_prefix = task_description + '\n' + input_prefix
    # print("---------- question generation ----------")
    # print("$$: "+final_prefix+" $$")
    return final_prefix

def answer_generation_input(data_item, TOKEN_MAXLEN, example_set):
    # task_description = "I am an intelligent bot that can play as judge in situation puzzles with user. A puzzle is given first, and the user begin to ask a \"yes/no\" question to ensure details, and I will give \"Yes/No/Irrelevent\" as answer to questions."
    task_description = "I can answer the question with only yes or no or irrelevant with the \"truth\". "
    puzzle, truth, fol_q, fol_a = extract_input_item(data_item)
    example_prefix = "truth:" + "".join(example["final_answer"]) + "\n"
    example_prefix += "follow_up_Q:"+example["question_list"][0]+ "\n" +"follow_up_A:"+example["answer_list"][0] + "\n"
    example_prefix += "follow_up_Q:"+example["question_list"][1]+ "\n" +"follow_up_A:"+example["answer_list"][1] + "\n"
    input_prefix = "truth:" + truth   
    if len(fol_q) and len(fol_a):
        for fq,fa in zip(fol_q, fol_a):
            input_prefix += "follow_up_Q:"+ fq
            input_prefix += "follow_up_A:"+ fa
    input_prefix += "\n"+"follow_up_Q:" + data_item["question_list"][-1] 
    input_prefix += "\n" + "follow_up_A:"
    example_start = example_truncation(example_prefix=example_prefix,input_prefix=input_prefix, TOKEN_MAXLEN=TOKEN_MAXLEN)
    if example_set==TRUE:
        final_prefix = task_description + '\n' + example_prefix[example_start:] + input_prefix
    else:
        final_prefix = task_description + '\n' + input_prefix
    # print("---------- answer generation ----------")
    # print("$$: "+final_prefix+" $$")
    return final_prefix

def solution_generate_input(data_item, TOKEN_MAXLEN, example_set):
    # task_description = "I am an intelligent bot that can play situation puzzles with user. A puzzle is given first, and the user begin to ask \"yes/no\" question to ensure details, then I will give the question a\"yes/no/irrelevent\" answer. Finally user try to give solution for the puzzle."
    task_description = "I can generate a logical answer to the question based on the puzzle, the follow_up_Q and the follow_up_Q. "
    puzzle, truth, fol_q, fol_a = extract_input_item(data_item)
    example_prefix = "puzzle:" + "".join(example["puzzle"]) + "\n"
    example_prefix += "follow_up_Q:"+example["question_list"][0]+"\n"+"follow_up_A:"+example["answer_list"][0] + "\n"
    example_prefix += "solution:" + "".join(example["final_answer"]) + "\n"
    input_prefix = "puzzle:" + puzzle + "\n"  
    if len(fol_q) and len(fol_a):
        for fq,fa in zip(fol_q, fol_a):
            input_prefix += "follow_up_Q:"+ fq
            input_prefix += "follow_up_A:"+ fa
    input_prefix += "\n" + "solution:"
    example_start = example_truncation(example_prefix=example_prefix,input_prefix=input_prefix, TOKEN_MAXLEN=TOKEN_MAXLEN)
    if example_set==TRUE:
        final_prefix = task_description + '\n' + example_prefix[example_start:] + input_prefix
    else:
        final_prefix = task_description + '\n' + input_prefix    
    # print("---------- solution generation ----------")
    # print("$$: "+final_prefix+" $$")
    return final_prefix

if __name__=="__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset_path', type=str, default="../data/merge_data.json")
    parse.add_argument('--sample_num', type=int, default=1)
    parse.add_argument('--chance_num', type=int, default=2)
    parse.add_argument('--threshold', type=int, default=0.85)
    parse.add_argument('--token_maxlen', type=int, default=700)
    parse.add_argument('--output_file', type=str, default="./output-e.txt")
    # parse.add_argument('--output_dataset', type=str, default="./new_dataset.json")
    # parse.add_argument('--with_hint', action='store_true')
    args = parse.parse_args()
    dataset = load_data(args.dataset_path)
    YNI =["Yes.", "No.", "Irrelevant."]
    # parse.add_argument('--with_hint', action='store_true')
    thinking_hint = ["correctly think!","reverse thinking!","think differently!",""]
    think_hint = ""
    for ditem in dataset[:args.sample_num]:
        print("PUZZLE:",ditem["puzzle"])
        chance_count = 0
        while chance_count < args.chance_num:
            # genrate question
            if ditem["answer_list"]:
                if ditem["answer_list"][-1] == "Yes.":
                    think_hint = thinking_hint[0]
                elif ditem["answer_list"][-1] == "No.":
                    think_hint = thinking_hint[1] 
                elif ditem["answer_list"][-1] == "irrelevant.":
                    think_hint = thinking_hint[2]
            if chance_count == 0:
                question_generation_prompt = question_generation_input(ditem,THINK_HINT=think_hint,TOKEN_MAXLEN=args.token_maxlen, example_set=TRUE)
            else:
                question_generation_prompt = question_generation_input(ditem,THINK_HINT=think_hint,TOKEN_MAXLEN=args.token_maxlen,example_set=FALSE)
            generated_question = get_response(question_generation_prompt)
            ditem["question_list"].append(generated_question)
            # generate answer
            waitforGPT()
            if chance_count == 0:
                answer_generation_prompt = answer_generation_input(ditem, TOKEN_MAXLEN=args.token_maxlen, example_set=TRUE)
            else:
                answer_generation_prompt = answer_generation_input(ditem, TOKEN_MAXLEN=args.token_maxlen, example_set=FALSE)
            generated_answer = get_response(answer_generation_prompt)
            if generated_answer in YNI:
                ditem["answer_list"].append(generated_answer)
                print("*",generated_question)
                print("*",generated_answer)
            else:
                ditem["question_list"].pop()
                continue
            waitforGPT()
            # generate solution
            if chance_count == 0:
                solution_generate_prompt = solution_generate_input(ditem, TOKEN_MAXLEN=args.token_maxlen, example_set=TRUE)
            else:
                solution_generate_prompt = solution_generate_input(ditem,TOKEN_MAXLEN=args.token_maxlen, example_set=FALSE)
            generated_solution = get_response(solution_generate_prompt)
            print("*",generated_solution)
            chance_count += 1
            # calculate the current solution's accuracy：Jaro-Winkler
            similarity_score = Levenshtein.jaro_winkler("".join(ditem["final_answer"]),generated_solution)
            print("the current similarity score is : ", similarity_score)
            if similarity_score >= args.threshold:
                break
            waitforGPT()
        with open(args.output_file,"a+",encoding="utf-8") as fp:
            result_out = "C " + str(chance_count) + "\n"
            result_out += "T " + "".join(ditem["final_answer"]) +"\n"
            result_out += "G " + generated_solution + "\n"
            fp.write(result_out)
            fp.close()