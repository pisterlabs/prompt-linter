from typing import List
import pandas as pd
import requests
from InstructorEmbedding import INSTRUCTOR
from src.utils import Prompt
import json
from sklearn.metrics.pairwise import cosine_similarity
from src.prompt_lib.prompt_lib.backends import openai_api
import re
import numpy as np
import ast
from pathlib import Path
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

from langchain import PromptTemplate,FewShotPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain,SequentialChain

from src.usmle.models.usmle_qtn_a import UsmleQtnAns
from src.usmle.models.usmle_whl_qtn import UsmleWholeQtn
from src.usmle.models.usmle_distr import UsmleDistr
from src.usmle.gen_order import gen_order
from dotenv import load_dotenv
import os
load_dotenv(dotenv_path='usmle.env')

OPENAIKEY = os.getenv("OPENAIKEY")
OPENAIORG = os.getenv("OPENAIORG")

print(OPENAIKEY)

class UsmleQgenTaskInitLgc(Prompt):
    def __init__(self, prompt_examples: str, engine: str) -> None:
        super().__init__(
            question_prefix="",
            answer_prefix="Context: ",
            intra_example_sep="\n\n",
            inter_example_sep="\n\n###\n\n",
        )
        self.engine = engine
        #self.setup_prompt_from_examples_file(prompt_examples)
    def retrieve_sim_questions_colbert(self,content,k:int):
        print(OPENAIKEY)
        p = {'query':content, 'k':k }
        r = requests.get( 'https://bio-nlp.org/colbertmimic/api/search',params=p)
        res_json = r.json()
        print(res_json)
        qbank_df = pd.read_csv('data/inputs/USMLE_qbank_whlqtn.csv')

        qbank_topk = res_json['topk']
        for item in qbank_topk:
            id = int(item['pid']) +1
            item['question'] = qbank_df.loc[id].at['question']
            item['correct_answer'] = qbank_df.loc[id].at['correct_answer']
            item['distractor_options'] = qbank_df.loc[id].at['distractor_options']
            item.pop('pid')
            item.pop('prob')
            item.pop('rank')
            item.pop('score')
        print(qbank_topk)
        return qbank_topk  
        
    def retrieve_sim_questions(self,content,instruction, k:int):
        model = INSTRUCTOR('hkunlp/instructor-large')
        embedding = model.encode([[instruction,content]])
        inputs_file_path = "data/US/US_qbank_embed.csv"
        qbank_df = pd.read_csv(inputs_file_path)
        qbank_df['embedding'] = qbank_df.apply(lambda x: self.get_embed_array(x),axis=1)
        qbank_df['cos_sim'] = qbank_df.apply(lambda x: cosine_similarity((x['embedding']),embedding),axis =1)
        qbank_df = qbank_df.sort_values(by=['cos_sim'], ascending=False)
        qbank_topk = qbank_df[:k].to_dict('records')
        for item in qbank_topk:
            item.pop('embedding')
            item.pop('cos_sim')
        return qbank_topk 
    def get_embed_array(self,row):
        x = row['embedding']
        x = re.sub("\[\s+", "[", x.strip())
        x = re.sub("\s+", ",", x.strip())
        x = ast.literal_eval(x)
        return x
    def generate_step_by_step(self,clinical_note: str,keypoint:str,topic:str) -> str:
        llm = ChatOpenAI(model=self.engine, temperature=0.7,openai_api_key=OPENAIKEY,openai_organization=OPENAIORG)
        ## Retrieving USMLE questions similar to the clinical note and topic from the qbank
        cn_sim_instruction = "Represent the following clinical note for clustering, for retrieving USMLE questions related to the topic '{topic}': "
        cn_sim_examples = self.retrieve_sim_questions_colbert(clinical_note, 3)
        print(cn_sim_examples)
        cn_sim_ex_prompt = PromptTemplate(
            suffix = "USMLE context based questions: \n",
            input_variables=["question"], 
            template="Context and Question: {question}")
        #print(qa_sim_ex_prompt.format())
        context_prompt = FewShotPromptTemplate(
            examples=cn_sim_examples, 
            example_prompt=cn_sim_ex_prompt, 
            suffix="Generate a context(not the question,in the format Context: ) based on the given topic from the clinical note :\nClinical Note: {clinical_note}\nTopic: {topic}\nKeypoint: {keypoint}", 
            input_variables=["clinical_note","topic","keypoint"])
            
        context_chain = LLMChain(llm=llm, prompt=context_prompt)
        context_output = context_chain.run({"clinical_note" : clinical_note,
            "keypoint" : keypoint,
            "topic" : topic})
        print(context_output)
        context = context_output.split('Context:')[1].strip()
        context_sim_instruction = "Represent the following context for clustering, for retrieving USMLE questions related to the topic '{topic}': "
        context_sim_examples = self.retrieve_sim_questions_colbert(context,  3)
        print(context_sim_examples)
        context_sim_ex_prompt = PromptTemplate(
            suffix = "USMLE context based questions: \n",
            input_variables=["question"], 
            template="Context and Question: {question}")
        #print(qa_sim_ex_prompt.format())
        question_prompt = FewShotPromptTemplate(
            examples=context_sim_examples, 
            example_prompt=context_sim_ex_prompt, 
            suffix="Generate a one line question(in the format Question: ) based on the given context:\nContext: {context}\nTopic: {topic}\nKeypoint: {keypoint}", 
            input_variables=["context","topic","keypoint"]
        )
        question_chain = LLMChain(llm=llm, prompt=question_prompt)
        question_output = question_chain.run({
            "context" : context,
            "keypoint" : keypoint,
            "topic" : topic
        })
        question = question_output.split('Question:')[1].strip()
        context_question = context+question
        cq_sim_instruction =  "Represent the following context and question for clustering, for retrieving USMLE questions related to the topic '{topic}': "
        cq_sim_examples = self.retrieve_sim_questions_colbert(context_question,  3)
        print(cq_sim_examples)
        cq_sim_ex_prompt = PromptTemplate(
            suffix = "USMLE context based questions with their correct answers: \n",
            input_variables=["question","correct_answer"], 
            template="Context and Question: {question}\nCorrect answer: {correct_answer}")
        #print(qa_sim_ex_prompt.format())
        ca_prompt = FewShotPromptTemplate(
            examples=cq_sim_examples, 
            example_prompt=cq_sim_ex_prompt, 
            suffix="Generate the correct answer(in the format Correct answer: ) to the question based on the given context,topic and keypoint(to which it should be highly related to) :\nContext: {context}\n Question: {question}\nTopic: {topic}\nKeypoint: {keypoint}", 
            input_variables=["context","question","keypoint","topic"]
        )
        ca_chain = LLMChain(llm=llm, prompt=ca_prompt)
        ca_output = ca_chain.run({
            "context" : context,
            "question" : question,
            "keypoint" : keypoint,
            "topic" : topic
        })
        correct_answer = ca_output.split('Correct answer:')[1].strip()
        qa_templ = "Question: {context_question}\nCorrect answer: {correct_answer}" 
        distr_instruction = "Represent the following context based question and its answer for clustering, for retrieving USMLE questions related to the topic '{topic}':"
        distr_sim_examples = self.retrieve_sim_questions_colbert(qa_templ.format(context_question=context_question,correct_answer=correct_answer),  3)
        #print(distr_sim_examples)
        distr_sim_ex_prompt = PromptTemplate(
            suffix = "USMLE context based questions with their correct answers and distractor options: \n",
            input_variables=["question", "correct_answer","distractor_options"], 
            template="Context and Question: {question}\nCorrect answer: {correct_answer}\nDistractor options: {distractor_options}")
        #print(distr_sim_ex_prompt.format())
        distr_prompt = FewShotPromptTemplate(
            examples=distr_sim_examples, 
            example_prompt=distr_sim_ex_prompt, 
            suffix="Generate distractor options(in the format Distractor options: ) for the context, question, and correct answer:\nContext: {context}\nQuestion: {question}\nCorrect answer: {correct_answer}", 
            input_variables=["context","question","correct_answer"]
        )
        distr_chain = LLMChain(llm=llm, prompt=distr_prompt)
        distr_output = distr_chain.run({
            "context" : context,
            "question" : question,
            "correct_answer" : correct_answer
        }
        )
        #print(distr_output)
        distractor_options = distr_output.split('Distractor options:')[1].strip()
        print(f"Context: {context}\nQuestion: {question}\nCorrect answer: {correct_answer}\nDistractor options:{distractor_options}")
        return context,question,correct_answer,distractor_options
    def generate_whole_qtn(self, clinical_note: str,keypoint:str,topic:str) -> str:
        llm = ChatOpenAI(model=self.engine, temperature=0.7,openai_api_key=OPENAIKEY,openai_organization=OPENAIORG)
        sim_examples = self.retrieve_sim_questions_colbert(clinical_note,  3)
        print(sim_examples)
        sim_ex_prompt = PromptTemplate(
            suffix = "USMLE context based questions with their correct answers and options: \n",
            input_variables=["question", "correct_answer","distractor_options"], 
            template="Context and Question: {question}\nCorrect answer: {correct_answer}\nDistractor options: {distractor_options}")
        parser = PydanticOutputParser(pydantic_object=UsmleWholeQtn)
        format_instructions = parser.get_format_instructions()
        prompt = FewShotPromptTemplate(
            examples=sim_examples, 
            example_prompt=sim_ex_prompt, 
            suffix="Generate a context and question(each separately), correct answer and distractor options for the following:\n{format_instructions}\nClinical Note: {clinical_note}\nTopic: {topic}\nKeypoint: {keypoint}", 
            input_variables=["clinical_note","topic","keypoint"],
            partial_variables={"format_instructions":  format_instructions}
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        output = chain.run({
            "clinical_note" : clinical_note,
            "keypoint" : keypoint,
            "topic" : topic
        })
        #print(ast.literal_eval(output)["context"])
        qa_output = ast.literal_eval(output)
        return qa_output['context'],qa_output['question'],qa_output['correct_answer'],qa_output['distractor_options']
    def generate_distractor_last(self, clinical_note: str,keypoint:str,topic:str) -> str:
        llm = ChatOpenAI(model=self.engine, temperature=0.7,openai_api_key=OPENAIKEY,openai_organization=OPENAIORG)
        qa_sim_instruction = "Represent the following clinical note for clustering, for retrieving USMLE questions related to the topic '{topic}': "
        qa_sim_examples = self.retrieve_sim_questions_colbert(clinical_note,  3)
        #print(qa_sim_examples)
        qa_sim_ex_prompt = PromptTemplate(
            suffix = "USMLE context based questions with their correct answers: \n",
            input_variables=["question", "correct_answer"], 
            template="Context and Question: {question}\nCorrect answer: {correct_answer}")
        #print(qa_sim_ex_prompt.format())
        qa_parser = PydanticOutputParser(pydantic_object=UsmleQtnAns)
        qa_format_instructions = qa_parser.get_format_instructions()
        qa_prompt = FewShotPromptTemplate(
            examples=qa_sim_examples, 
            example_prompt=qa_sim_ex_prompt, 
            suffix="Generate a context and question(each separately) and correct answer:\n{format_instructions}\nClinical Note: {clinical_note}\nTopic: {topic}\nKeypoint: {keypoint}", 
            input_variables=["clinical_note","topic","keypoint"],
            partial_variables={"format_instructions":  qa_format_instructions}
        )
        qa_chain = LLMChain(llm=llm, prompt=qa_prompt)
        qa_output = qa_chain.run({
            "clinical_note" : clinical_note,
            "keypoint" : keypoint,
            "topic" : topic
        })
        qa_output = ast.literal_eval(qa_output)
        print(qa_output.keys())
        context,question,correct_answer = qa_output["context"],qa_output["question"],qa_output["correct_answer"]
        distr_instruction = "Generate distractor options for the following context, question and correct answers."
        distr_sim_examples = self.retrieve_sim_questions_colbert(str(qa_output)[1:-1], 3)
        #print(distr_sim_examples)
        distr_sim_ex_prompt = PromptTemplate(
            suffix = "USMLE context based questions with their correct answers and distractor options: \n",
            input_variables=["question", "correct_answer","distractor_options"], 
            template="Context and Question: {question}\nCorrect answer: {correct_answer}\nDistractor options: {distractor_options}")
        #print(distr_sim_ex_prompt.format())
        response_schemas = [
            ResponseSchema(name="distractor_options", description="Distractor options which are incorrect but make sense given the context and question.")
        ]
        # distr_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        # distr_instructions = distr_parser.get_format_instructions()
        distr_parser = PydanticOutputParser(pydantic_object=UsmleDistr)
        distr_instructions = distr_parser.get_format_instructions()
        distr_prompt = FewShotPromptTemplate(
            examples=distr_sim_examples, 
            example_prompt=distr_sim_ex_prompt, 
            suffix="Generate distractor options for the context, question, and correct answer:\n{format_instructions}\nContext: {context}\nQuestion: {question}\nCorrect answer: {correct_answer}", 
            input_variables=["context","question","correct_answer"],
            partial_variables={"format_instructions": distr_instructions}
        )
        distr_chain = LLMChain(llm=llm, prompt=distr_prompt)
        distr_output = distr_chain.run({
            "context" : context,
            "question" : question,
            "correct_answer" : correct_answer
        })
        #print(distr_output)
        distractor_options = distr_output.split(':')[1].strip()
        #print(distractor_options)
        qa_output['distractor_options'] = distractor_options
        return qa_output['context'],qa_output['question'],qa_output['correct_answer'],qa_output['distractor_options']
    def __call__(self, clinical_note: str,keypoint:str,topic:str,order_enum:gen_order) -> str:
        match order_enum:
            case gen_order.tandem:
                return self.generate_whole_qtn(clinical_note,keypoint,topic)
            case gen_order.distractor_options_last:
                return self.generate_distractor_last(clinical_note,keypoint,topic)
            case gen_order.step_by_step:
                return self.generate_step_by_step(clinical_note,keypoint,topic)

if __name__ == "__main__":
    task_init = UsmleQgenTaskInitLgc(
        prompt_examples="data/prompt/usmle/init.jsonl",
        engine="gpt-3.5-turbo"
    )
    
    #print(task_init.prompt)
    cn = "An 84-year-old female with a past medical history of hypertension presented with weakness, dry cough, and shortness of breath for four days. The patient had received two doses of the COVID vaccine, with the second dose in March 2021. In the ER, her vital signs were BP 133/93, HR 103 bpm, RR 22 breaths/min, oxygen saturation of 96% on 40 L per minute of supplemental oxygen via high-flow nasal cannula, and afebrile. Laboratory assessment is in Table. Nasopharyngeal swab for SARS-CoV-2 RNA was positive. Chest X-ray on admission shows worsening right pleural effusion with new opacity obscuring the lower two-third of the right lung and a new pleural-based opacity in the left upper lobe Figure. CT chest with contrast shows large right pleural effusion and associated right basilar consolidation and abdominal ascites. The patient was admitted to the telemetry unit and started on methylprednisolone, piperacillin-tazobactam, remdesivir, and baricitinib. The patient clinically deteriorated on Day 2 and was transferred to the intensive care unit for thoracentesis and possible intubation. Thoracentesis removed 1.95 L of bloody, serosanguineous fluid obtained, with partial resolution of the effusion Figure. On Day 3, the patient developed septic shock, florid renal failure, and lethargy and was started on intravenous fluids and norepinephrine drip. Chest X-ray showed near-complete opacification of bilateral lung fields and subsequently was intubated. On Day 4, tense ascites were noted and the patient underwent paracentesis, which removed 4.25 L of bloody, serosanguinous fluid. Renal replacement therapy started. The patient was deemed to have a guarded prognosis with multiorgan failure."
    topic = "underlying processes/pathways"
    keypoint = "pathophysiology of sepsis"

    task_init.__call__(clinical_note=cn,keypoint=keypoint,topic=topic)