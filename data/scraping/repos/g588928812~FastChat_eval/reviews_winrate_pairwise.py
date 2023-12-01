import argparse
import json
import os
import time

import requests
import openai
import tqdm
import re

import shortuuid
import logging
import glob

from datetime import datetime

logging.basicConfig(level=logging.INFO)

logFormatter = logging.Formatter("%(asctime)s %(message)s")
logger = logging.getLogger()

def get_json_list_dir(dir_path):
    files = glob.glob(os.path.expanduser(dir_path) + "/*.jsonl")

    jsonl=[]
    for f in files:
        jsonl.extend(get_json_list(f))

    return jsonl

def get_json_list(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, "r") as f:
        json_list = []
        for line in f:
            json_list.append(json.loads(line, strict=False))
        return json_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChatGPT-based QA evaluation.")
    parser.add_argument("-r", "--review-dir", default="table/reviews_pairwise")
    parser.add_argument("-a", "--answer-dir", default="table/answer")
    parser.add_argument("-e", "--exclude-math-code-questions", type=bool, default=True)
    args = parser.parse_args()

    assert(args.review_dir != None and args.answer_dir != None)

    reviews_json = get_json_list_dir(args.review_dir)
    answers_json = get_json_list_dir(args.answer_dir)

    if args.exclude_math_code_questions:
        print("Excluding math questions")
        questions_exclude=[61, 62, 63, 64, 65, 66, 67, 68, 69, 70]
    else:
        questions_exclude=[]

    model_reviews=[]
    for i in range(len(reviews_json)):      # assign reviews to each model, check if there are replicates
        review=reviews_json[i]

        if review["question_id"] in questions_exclude:
            continue

        answers=[]
        answers.append(next(item for item in answers_json if item["answer_id"] == review["answer1_id"]))
        answers.append(next(item for item in answers_json if item["answer_id"] == review["answer2_id"]))
        scores = review["scores"]
        models = [answers[0]["model_id"], answers[1]["model_id"]]

        for f in [0,1]:
            model=models[f]
            ans1=answers[0]
            ans2=answers[1]
            answer_pair=answers[0]["answer_id"]+answers[1]["answer_id"]

            model_reviews_found = list(filter(lambda model_reviews: model_reviews["model_id"] == model, model_reviews))

            if len(model_reviews_found)==0:
                model_review={
                    "model_id": model,
                    "replicates": [[], [], [], [], []]     # array of arrays of reviews replicate[0]=array of reviews replicate 1
                }
                model_reviews.append(model_review)
            else:
                model_review=model_reviews_found[0]

            if len(model_review["replicates"])==0:
                model_review["replicates"].append([review])
            else:
                for g in range(len(model_review["replicates"])):                
                    replicates=model_review["replicates"][g]
                    review_found=None
                    for r in replicates:
                        if r["answer1_id"]==ans1["answer_id"] and r["answer2_id"]==ans2["answer_id"]:
                            review_found=r
                    if review_found is None:
                        replicates.append(review)
                        break

    # Sanity check: for each model how many replicates and each other model
    for i in range(len(model_reviews)):      
        model=model_reviews[i]["model_id"]
        replicates=model_reviews[i]["replicates"]

        for replicateNo in range(len(replicates)):    
            replicate=replicates[replicateNo]
            counts={}
            for review in replicate:
                answers=[]
                answers.append(next(item for item in answers_json if item["answer_id"] == review["answer1_id"]))
                answers.append(next(item for item in answers_json if item["answer_id"] == review["answer2_id"]))
                if answers[0]["model_id"]==model:
                    othermodel=answers[1]["model_id"]
                else:
                    othermodel=answers[0]["model_id"]
                if not othermodel in counts:
                    counts[othermodel]=0
                counts[othermodel]+=1
            for othermodel in counts:
                print("{}\trep{}\t{}\t{}".format(model, replicateNo,othermodel, counts[othermodel]))  

    # winrate for each model and replicate
    print("\nWINRATES")
    for i in range(len(model_reviews)):      
        model=model_reviews[i]["model_id"]
        replicates=model_reviews[i]["replicates"]

        for replicateNo in range(len(replicates)):    
            replicate=replicates[replicateNo]
            if len(replicate)==0:
                continue

            scores={"wins":0, "ties":0, "notParsed":0, "overall":0}        
            for review in replicate:
                winner=review["winner"]
                answers=[]
                answers.append(next(item for item in answers_json if item["answer_id"] == review["answer1_id"]))
                answers.append(next(item for item in answers_json if item["answer_id"] == review["answer2_id"]))
                model1=answers[0]["model_id"]
                model2=answers[1]["model_id"]
                winner=review["winner"]

                if winner==0:
                    scores["ties"]+=1
                elif (winner==1 and model1==model):
                    scores["wins"]+=1
                elif (winner==2 and model2==model):
                    scores["wins"]+=1
                elif winner==-1:
                    scores["notParsed"]+=1

                scores["overall"]+=1

            print("{model}\treplicate{rep}\t{rate}".format(
                model=model,
                rep=replicateNo,
                rate=round((2*scores["wins"]+scores["ties"])/(2*scores["overall"])*100,3),
                q=len(scores)
                ))
            # print("Winner could not be determined in {} cases".format(scores["notParsed"]))

