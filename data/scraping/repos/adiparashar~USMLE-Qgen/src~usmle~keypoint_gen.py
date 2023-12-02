import json
import pathlib
import random
from tqdm import tqdm
from typing import List
import re
import requests
from src.usmle.task_init import UsmleQgenTaskInit
from src.usmle.task_iterate import UsmleQgenTaskIterate
from src.usmle.feedback import UsmleQgenFeedback
from src.usmle.answer import UsmleQgenAnswer
from src.utils import retry_parse_fail_prone_cmd
import ast
from src.usmle.feedback_lgc import UsmleQgenFeedbackLgc
from src.usmle.gen_order import gen_order

from langchain import PromptTemplate,FewShotPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain,SequentialChain
from src.usmle.task_init_lgc import UsmleQgenTaskInitLgc
from dotenv import load_dotenv
import os
load_dotenv(dotenv_path='usmle.env')

OPENAIKEY = os.getenv("OPENAIKEY")
OPENAIORG = os.getenv("OPENAIORG")

CODEX = "code-davinci-002"
GPT3 = "text-davinci-003"
CHATGPT = "gpt-3.5-turbo"
ENGINE = "gpt-4"

@retry_parse_fail_prone_cmd
# def usmlekpgen(self,engine: str) -> str:
#     self.engine = ENGINE
#     return content_to_fb_ret
def retrieve_sim_keypoints_colbert(content,k:int):
        p = {'query':content, 'k':k }
        r = requests.get( 'https://bio-nlp.org/colbertmimic/api/search',params=p)
        res_json = r.json()
        # print(res_json)
        qbank_topk = res_json['topk'][0]
        print(qbank_topk)
        return qbank_topk  
def gen_keypoints(clinical_note:str,topic:str,examples_path:str):
    llm = ChatOpenAI(model=ENGINE, temperature=0.7,openai_api_key=OPENAIKEY,openai_organization=OPENAIORG)
    ## Retrieving USMLE questions similar to the clinical note and topic from the qbank
    retrieved_kp = retrieve_sim_keypoints_colbert(clinical_note, 1)
    retrieved_kp_score = retrieved_kp['score']
    retrieved_keypoints = retrieved_kp['text']
    #print(kp_sim_examples)
    with open(examples_path) as f:
        fs_dict = json.load(f)
    print(fs_dict)
    kp_sim_ex_prompt = PromptTemplate(
        suffix = "Example keypoints for a given clinical note and based on a topic: ",
        input_variables=["clinical_note","topic","keypoint"], 
        template="Clinical note: {clinical_note}\nTopic: {topic}\nKeypoint: {keypoint}")
    #  print(kp_sim_ex_prompt.format(clinical_note=clinical_note,topic=topic,))
    keypoint_prompt = FewShotPromptTemplate(
        examples=fs_dict, 
        example_prompt=kp_sim_ex_prompt, 
        suffix="Please extract a keypoint from the provided list of USMLE concepts. These concepts are organized in a hierarchical manner, starting from the most general and progressively becoming more specific. The keypoint you extract should ideally be specific and concise, covering one or two USMLE concepts. This keypoint will be used as the central focus for generating a USMLE question based on a clinical note within the specified topic. The goal is to ensure a strong and relevant connection between the concept and the question..:\nClinical Note: {clinical_note}\nTopic: {topic}\n USMLE concepts: {usmle_concepts}\nKeypoint:", 
        input_variables=["clinical_note","topic","usmle_concepts"])
    
    keypoint_chain = LLMChain(llm=llm, prompt=keypoint_prompt)
    keypoint_output = keypoint_chain.run({"clinical_note" : clinical_note,
        "topic" : topic,
        "usmle_concepts" : retrieved_keypoints})
    print(keypoint_output)
    return retrieved_kp_score,retrieved_keypoints, keypoint_output

def run_cmd():
    concepts = sys.argv[2:]
    max_attempts = 5
    content_to_fb = autofb_usmleqgen(
        concepts=concepts,
        max_attempts=max_attempts,
    )

    res = []
    for s in  content_to_fb:
        sent = s["sentence"]
        fb = ";  ".join(s["concept_feedback"]) + " " + s["commonsense_feedback"]
        res.append(f"{sent} ({fb})")
    print(" -> ".join(res))


def run_iter(inputs_file_path: str, examples_path:str, max_attempts: int = 4):
    print(inputs_file_path)
    test_df = pd.read_json(inputs_file_path,lines=True, orient="records")
    # test_df = test_df.sample(10)
    test_df = test_df[581:]
    print(test_df.columns)
    # add new columns  content_to_fb of type object, and status of type string

    is_rerun = "status" in test_df.columns
    if not is_rerun:
        test_df["keypoint_data"] = None
        test_df["keypoint_data"] = test_df["keypoint_data"].astype(object)
        test_df["status"] = None
        #this is a test comment

    else:
        print("Status column already exists! Looks like you're trying to do a re-run")
        print(test_df["status"].value_counts())
    for i, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Running autofb iter"):
        if row["status"] == "success":
            continue
        # try:
        topic_list = ast.literal_eval(row['topic_list'])
        print(topic_list)
        for topic in topic_list:
            retrieved_kp_score,retrieved_keypoints, keypoint_output = gen_keypoints(clinical_note=row["clinical_note"],topic=topic,examples_path=examples_path)
            #autofb_usmleqgen(clinical_note=row["clinical_note"],keypoint=row['keypoint'],topic=row['topic'], max_attempts=max_attempts)
            keypoint_data = {"keypoint_data" : {"retrieved_kp_score":retrieved_kp_score,"retrieved_keypoints":retrieved_keypoints,"keypoint_gpt4_output":keypoint_output}}

            # except Exception as e:
            #     keypoint_data = "Some error occured: " + str(e)
            dict_to_write = {"clinical_note":row["clinical_note"],"topic":topic, "keypoint_data" : keypoint_data}
            output_path = inputs_file_path + (".iter.out" if not is_rerun else ".v0")
            version = 1
            # while pathlib.Path(output_path).exists():
            output_path = output_path + f".v{version}"
            #     version += 1
            with open(output_path, 'a+') as f:
                json.dump(dict_to_write,f)
                f.write('\n')
            print(f"keypoint_data : {keypoint_data}")
            test_df.at[i, "keypoint_data"] = keypoint_data
            test_df.at[i, "status"] = "success"
        # except Exception as e:
        #     test_df.loc[i, "content_to_fb"] = str(e)
        #     test_df.loc[i, "status"] = "error"

    output_path = inputs_file_path + (".iter.out" if not is_rerun else ".v0")
    version = 0
    while pathlib.Path(output_path).exists():
        output_path = output_path + f".v{version}"
        version += 1

    test_df.to_json(output_path, orient="records", lines=True)


def run_multi_sample(self,inputs_file_path: str, n_samples: int = 4):
    # print(inputs_file_path)
    #this should work
    test_df = pd.read_json(inputs_file_path, lines=True, orient="records")

    is_rerun = "status" in test_df.columns
    if not is_rerun:
        test_df["outputs"] = None
        test_df["outputs"] = test_df["outputs"].astype(object)
        test_df["status"] = None 
    else:
        print("Status column already exists! Looks like you're trying to do a re-run")
        print(test_df["status"].value_counts())

    task_init = UsmleQgenTaskInit(engine=ENGINE, prompt_examples="data/prompt/usmle/init.jsonl")
    task_feedback = UsmleQgenFeedback(
        engine=ENGINE, prompt_examples="data/prompt/usmle/feedback.jsonl"
    )
    for i, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Running multisample autofb"):

        if row["status"] == "success":
            continue
        try:
            outputs = []
            for _ in range(n_samples):
                sent = task_init(concepts=row["concepts"])
                print(sent)
                concept_fb, commonsense_fb = task_feedback(concepts=row["concepts"], sentence=sent)
                print(concept_fb, commonsense_fb)
                outputs.append(
                    {
                        "sentence": sent,
                        "concept_feedback": [f.strip() for f in concept_fb.split(",")],
                        "commonsense_feedback": commonsense_fb,
                    }
                )
                if concept_fb.lower() == "none" and commonsense_fb.lower() == "none":
                    break
            test_df.loc[i, "outputs"] = outputs
            test_df.loc[i, "status"] = "success"
        except Exception as e:
            raise e
            test_df.loc[i, "outputs"] = str(e)
            test_df.loc[i, "status"] = "error"
    print(test_df)
    output_path = inputs_file_path + "." + ENGINE + (".multi.out" if not is_rerun else ".v0")
    version = 0
    while pathlib.Path(output_path).exists():
        output_path = output_path + f".v{version}"
        version += 1

    test_df.to_json(output_path, orient="records", lines=True)


if __name__ == "__main__":
    import sys
    import pandas as pd

    if sys.argv[1] == "cmd":
        run_cmd()

    elif sys.argv[1] == "batch-iter":
        run_iter(inputs_file_path=sys.argv[2],examples_path = sys.argv[3])

    elif sys.argv[1] == "batch-multi":
        run_multi_sample(inputs_file_path=sys.argv[2])

    else:
        raise ValueError("Invalid mode: choose between cmd, batch-iter, batch-multi")


