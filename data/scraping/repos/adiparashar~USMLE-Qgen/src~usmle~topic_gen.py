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
CODEX = "code-davinci-002"
GPT3 = "text-davinci-003"
CHATGPT = "gpt-3.5-turbo"
ENGINE = "gpt-4"
from dotenv import load_dotenv
import os
load_dotenv(dotenv_path='usmle.env')

OPENAIKEY = os.getenv("OPENAIKEY")
OPENAIORG = os.getenv("OPENAIORG")
@retry_parse_fail_prone_cmd
# def usmlekpgen(self,engine: str) -> str:
#     self.engine = ENGINE
#     return content_to_fb_ret
def gen_topics(clinical_note:str,examples_path:str):
    llm = ChatOpenAI(model=ENGINE, temperature=0.7,openai_api_key=OPENAIKEY,openai_organization=OPENAIORG)
    with open(examples_path) as f:
        fs_dict = json.load(f)
    print(fs_dict)
    topic_sim_ex_prompt = PromptTemplate(
        suffix = "Clinical notes and their associated topics",
        input_variables=["clinical_note","topic_list"], 
        template="Clinical note: {clinical_note}\nTopic list: {topic_list}")
    #  print(kp_sim_ex_prompt.format(clinical_note=clinical_note,topic=topic,))
    topic_prompt = FewShotPromptTemplate(
        examples=fs_dict, 
        example_prompt=topic_sim_ex_prompt, 
        suffix="Please select five (5) topics from the provided list of USMLE topics that are closely related to the given clinical note, use the clinical topic and topic list examples as a reference. These topics should be suitable for creating USMLE context-based questions that align with the content of the clinical notes. \nClinical Note: {clinical_note}\n USMLE topics: {usmle_topics}\nTopic list:", 
        input_variables=["clinical_note","usmle_topics"])
    usmle_topics = [
    "the cause/infectious agent or predisposing factor(s)",
    "underlying processes/pathways",
    "underlying anatomic structure or physical location",
    "mechanisms, drugs",
    "knows signs/symptoms of selected disorders",
    "knows individual’s risk factors for development of condition",
    "knows what to ask to obtain pertinent additional history",
    "predicts the most likely additional physical finding",
    "select most appropriate laboratory or diagnostic study",
    "interprets laboratory or other study findings",
    "predicts the most likely laboratory or diagnostic study result",
    "most appropriate laboratory or diagnostic study after change in patient status",
    "select most likely diagnosis",
    "recognizes factors in the history, or physical or laboratory study findings",
    "interprets laboratory or other diagnostic study results and identifies current/future status of patient",
    "recognizes associated conditions of a disease",
    "recognizes characteristics of disease relating to natural history or course of disease",
    "risk factors for conditions amenable to prevention or detection",
    "identifies patient groups at risk",
    "knows common screening tests",
    "selects appropriate preventive agent or technique",
    "knows appropriate counseling regarding current and future problems",
    "educates patients",
    "selects most appropriate pharmacotherapy",
    "assesses patient adherence, recognizes techniques to increase adherence",
    "recognizes factors that alter drug requirements",
    "Knows adverse effects of various drugs or recognizes signs and symptoms of drug (and drug-drug) interactions",
    "knows contraindications of various medications",
    "knows modifications of a therapeutic regimen within the context of continuing care",
    "appropriate monitoring to evaluate effectiveness of pharmacotherapy or adverse effects",
    "most appropriate management of selected conditions",
    "immediate management or priority in management",
    "follow-up or monitoring approach regarding the management plan",
    "current/short-term management",
    "severity of patient condition in terms of need for referral for surgical treatments/procedures",
    "appropriate surgical management",
    "preoperative/postoperative",
    "Selecting Clinical Interventions (Mixed Management)",
    "indications for surveillance for recurrence or progression of disease following treatment",
    "how to monitor a chronic disease in a stable patient where a change in patient status might indicate a need to change therapy",
    "most appropriate long-term treatment"
]

    topic_chain = LLMChain(llm=llm, prompt=topic_prompt)
    topic_output = topic_chain.run({"clinical_note" : clinical_note,
        "usmle_topics" : usmle_topics})
    print(topic_output)
    return topic_output

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
    test_df = pd.read_json(inputs_file_path, orient="records")
    # test_df = test_df.sample(50)
    print(test_df.columns)
    # add new columns  content_to_fb of type object, and status of type string

    is_rerun = "status" in test_df.columns
    if not is_rerun:
        test_df["topic_list"] = None
        test_df["topic_list"] = test_df["topic_list"].astype(object)
        test_df["status"] = None
        #this is a test comment

    else:
        print("Status column already exists! Looks like you're trying to do a re-run")
        print(test_df["status"].value_counts())
    for i, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Running autofb iter"):
        if row["status"] == "success":
            continue
        # try:
        topic_list = gen_topics(clinical_note=row["clinical_note"],examples_path=examples_path)
        #autofb_usmleqgen(clinical_note=row["clinical_note"],keypoint=row['keypoint'],topic=row['topic'], max_attempts=max_attempts)
        # keypoint_data = {"topic_list" : rieved_kp_score,"retrieved_keypoints":retrieved_keypoints,"keypoint_gpt4_output":keypoint_output}}
        
        # except Exception as e:
        #     keypoint_data = "Some error occured: " + str(e)
        dict_to_write = {"clinical_note":row["clinical_note"],"topic_list":topic_list}
        output_path = inputs_file_path + (".iter.out" if not is_rerun else ".v0")
        version = 1
        # while pathlib.Path(output_path).exists():
        output_path = output_path + f".v{version}"
        #     version += 1
        with open(output_path, 'a+') as f:
            json.dump(dict_to_write,f)
            f.write('\n')
        print(f"topic_list : {topic_list}")
        test_df.at[i, "topic_list"] = topic_list
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
    task_init = UsmleQgenTaskInitLgc(
        prompt_examples="data/prompt/usmle/init.jsonl",
        engine="gpt-3.5-turbo"
    )
    
    #print(task_init.prompt)
    cn = "An 84-year-old female with a past medical history of hypertension presented with weakness, dry cough, and shortness of breath for four days. The patient had received two doses of the COVID vaccine, with the second dose in March 2021. In the ER, her vital signs were BP 133/93, HR 103 bpm, RR 22 breaths/min, oxygen saturation of 96% on 40 L per minute of supplemental oxygen via high-flow nasal cannula, and afebrile. Laboratory assessment is in Table. Nasopharyngeal swab for SARS-CoV-2 RNA was positive. Chest X-ray on admission shows worsening right pleural effusion with new opacity obscuring the lower two-third of the right lung and a new pleural-based opacity in the left upper lobe Figure. CT chest with contrast shows large right pleural effusion and associated right basilar consolidation and abdominal ascites. The patient was admitted to the telemetry unit and started on methylprednisolone, piperacillin-tazobactam, remdesivir, and baricitinib. The patient clinically deteriorated on Day 2 and was transferred to the intensive care unit for thoracentesis and possible intubation. Thoracentesis removed 1.95 L of bloody, serosanguineous fluid obtained, with partial resolution of the effusion Figure. On Day 3, the patient developed septic shock, florid renal failure, and lethargy and was started on intravenous fluids and norepinephrine drip. Chest X-ray showed near-complete opacification of bilateral lung fields and subsequently was intubated. On Day 4, tense ascites were noted and the patient underwent paracentesis, which removed 4.25 L of bloody, serosanguinous fluid. Renal replacement therapy started. The patient was deemed to have a guarded prognosis with multiorgan failure."
    # gen_topics(cn,'/Users/adityaparashar/BioNLP/USMLE-QG/data/prompt/usmle/topic_fewshot.jsonl')
    if sys.argv[1] == "cmd":
        run_cmd()

    elif sys.argv[1] == "batch-iter":
        run_iter(inputs_file_path=sys.argv[2],examples_path = sys.argv[3])

    elif sys.argv[1] == "batch-multi":
        run_multi_sample(inputs_file_path=sys.argv[2])

    else:
        raise ValueError("Invalid mode: choose between cmd, batch-iter, batch-multi")


