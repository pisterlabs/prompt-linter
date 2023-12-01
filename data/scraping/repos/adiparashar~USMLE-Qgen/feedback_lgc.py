import ast
import re
from typing import Set, List
import pandas as pd
from src.prompt_lib.prompt_lib.backends import openai_api
import json
from src.utils import Prompt
import langchain

from src.usmle.models.reasoning_feedback import reasoning_feedback
langchain.debug = True
from langchain.output_parsers import PydanticOutputParser
from src.usmle.models.context_feedback import context_feedback
from src.usmle.models.correct_answer_feedback import correct_answer_feedback
from src.usmle.models.distractor_options_feedback import distractor_options_feedback
from src.usmle.models.question_feedback import question_feedback
from langchain.prompts import FewShotPromptTemplate
from src.usmle.prompts.component_fb_prompt_template import ComponentFeedbackPromptTemplate
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv(dotenv_path='usmle.env')

OPENAIKEY = os.getenv("OPENAIKEY")
OPENAIORG = os.getenv("OPENAIORG")

class UsmleQgenFeedbackLgc(Prompt):
    def __init__(self, engine: str, prompt_examples: str,rubrics_path:str, max_tokens: int = 2048) -> None:
        super().__init__(
            question_prefix="",
            answer_prefix="",
            intra_example_sep="\n\n",
            inter_example_sep="\n\n###\n\n",
        )
        self.engine = engine
        self.max_tokens = max_tokens
        self.examples_path = prompt_examples
        self.rubrics_path = rubrics_path
        # self.setup_prompt_from_examples_file(prompt_examples,reasoning_rubrics)
        self.fb_dict = {
            "context":PydanticOutputParser(pydantic_object=context_feedback),
            "question":PydanticOutputParser(pydantic_object=question_feedback),
            "correct_answer":PydanticOutputParser(pydantic_object=correct_answer_feedback),
            "distractor_options":PydanticOutputParser(pydantic_object=distractor_options_feedback),
            "reasoning":PydanticOutputParser(pydantic_object=reasoning_feedback)
        }
    
    def setup_feedback(self,feedback):
        fb = ''
        for key in feedback.keys():
            fb += '\n' + key + ':' + feedback[key] 
        return fb
    def feedback_step_by_step(self,examples_path,rubrics_path,clinical_note: str, keypoint:str, topic:str, context:str, question:str, correct_answer:str, distractor_options:str,attempted_answer:str,reasoning:str):
        context_feedback = self.feedback_comp("context",examples_path,rubrics_path,clinical_note,keypoint, topic, context, question, correct_answer, distractor_options,attempted_answer,reasoning)
        question_feedback = self.feedback_comp("question",examples_path,rubrics_path,clinical_note,keypoint, topic, context, question, correct_answer, distractor_options,attempted_answer,reasoning)
        correct_answer_feedback = self.feedback_comp("correct_answer",examples_path,rubrics_path,clinical_note,keypoint, topic, context, question, correct_answer, distractor_options,attempted_answer,reasoning)
        distractor_options_feedback = self.feedback_comp("distractor_options",examples_path,rubrics_path,clinical_note,keypoint, topic, context, question, correct_answer, distractor_options,attempted_answer,reasoning)
        reasoning_feedback = self.feedback_reasoning("reasoning",examples_path,rubrics_path,clinical_note,keypoint, topic, context, question, correct_answer, distractor_options,attempted_answer,reasoning)
        
        return ast.literal_eval(context_feedback),ast.literal_eval(question_feedback),ast.literal_eval(correct_answer_feedback),ast.literal_eval(distractor_options_feedback),ast.literal_eval(reasoning_feedback)
    def feedback_comp(self,comp_name,examples_path,rubrics_path,clinical_note: str, keypoint:str, topic:str, context:str, question:str, correct_answer:str, distractor_options:str,attempted_answer:str,reasoning:str):
        PROMPT = """
            In addition to the scoring rubrics in the examples above,give feedback and score the {component_name} using the attempted answer's(correct/incorrect) reasoning-based rubrics and their definitions below.
            Please include both the previous scoring rubrics and the following reasoning-based rubrics before giving the feedback for a particular aspect and add up the scores for all the aspects for the total scores of the {component_name}. 
            Many of these feedback points for the {component_name} depend upon the reasoning and the attempted answer correctness so consider that while giving feedback for the {component_name}.
            {component_name} reasoning-based rubrics: {reasoning_rubrics}
            Give the output in just this format: {format_instructions}
            Output just the JSON instance and nothing else.
            Clinical note: {clinical_note}
            Keypoint: {keypoint}
            Topic: {topic}
            Context: {context}
            Question: {question}
            Correct answer: {correct_answer}
            Attempted answer: {attempted_answer}
            Reasoning: {reasoning}
            Distractor options: {distractor_options}
        """
        llm = ChatOpenAI(model=self.engine, temperature=0.7,openai_api_key=OPENAIKEY,openai_organization=OPENAIORG)
        with open(examples_path) as f:
            dictt = json.load(f)
        ex_comp_fb = {k:dictt[k] for k in dictt.keys() if k.lower().find(comp_name) != -1}
        ex_content = {k:dictt[k] for k in dictt.keys() if (k.lower().find('feedback') == -1 and k.lower().find('score') == -1)}
        feedback_key = [j for j in ex_comp_fb.keys() if j.lower().find('feedback') != -1][0]
        score_key = [j for j in ex_comp_fb.keys() if j.lower().find('score') != -1][0]
        ex_comp_fb['feedback'] = self.setup_feedback(ex_comp_fb[feedback_key])
        ex_comp_fb['score'] = ex_comp_fb[score_key]
        ex_comp_fb['component_name'] = comp_name
        del ex_comp_fb[score_key]
        del ex_comp_fb[feedback_key]
        ex_comp_fb.update(ex_content)
        ex_comp = [ex_comp_fb]
        with open(rubrics_path) as f:
            dictt = json.load(f)
        ex_comp_fb_prompt = PromptTemplate(
            suffix = "Example USMLE component generated from the clinical note and its respective feedback: \n",
            input_variables=["clinical_note", "topic","keypoint","context","question","correct_answer","distractor_options","component_name","feedback","score"], 
            template="Clinical note:{clinical_note}\nTopic: {topic}\nKeypoint: {keypoint}\nContext:{context}\nQuestion:{question}\nCorrect answer:{correct_answer}\nDistractor options:{distractor_options}\n{component_name} feedback: {feedback}\n{component_name} score: {score}")
        #print(qa_sim_ex_prompt.format())
        fb_parser = self.fb_dict[comp_name]
        fb_format_instructions = fb_parser.get_format_instructions()
        rub_comp_fb = {k:dictt[k] for k in dictt.keys() if k.lower().find(comp_name) != -1}
        rub_comp_fb[list(rub_comp_fb.keys())[0]] = self.setup_feedback(rub_comp_fb[list(rub_comp_fb.keys())[0]])
        fb_generator = FewShotPromptTemplate(
            examples=ex_comp,
            example_prompt=ex_comp_fb_prompt,
            suffix=PROMPT,
            input_variables=["component_name","clinical_note", "topic","keypoint","context","question","correct_answer","attempted_answer","reasoning","distractor_options","reasoning_rubrics"],
            partial_variables={"format_instructions": fb_format_instructions})
        fb_generator.examples =ex_comp
        fb_chain = LLMChain(llm=llm, prompt=fb_generator)
        fb_output = fb_chain.run({
            "component_name":comp_name,
            "clinical_note" : clinical_note,
            "keypoint" : keypoint,
            "topic" : topic,
            "context" : context,
            "question" : question,
            "correct_answer" : correct_answer,
            "distractor_options" : distractor_options,
            "attempted_answer":attempted_answer,
            "reasoning":reasoning,
            "reasoning_rubrics": list(rub_comp_fb.values())[0]            
        })
        print(fb_output)
        return fb_output
    def feedback_reasoning(self,comp_name,examples_path,rubrics_path,clinical_note: str, keypoint:str, topic:str, context:str, question:str, correct_answer:str, distractor_options:str,attempted_answer:str,reasoning:str):
        PROMPT = """
            Give supporting textual feedback for each aspect and score(out of 5 for each aspect, in the format "2/5" if the score for that aspect is 2, also give supporting evidence for that score) the {component_name} using the attempted answer's(correct/incorrect) reasoning-based rubrics and their definitions below.
            Please include the following reasoning-based rubrics before giving the feedback for a particular aspect and add up the scores for all the aspects for the total score of the {component_name}. 
            Many of these feedback points for the {component_name} depend upon the reasoning and the attempted answer correctness so consider that while giving feedback for the {component_name}.
            {component_name} rubrics: {reasoning_rubrics}
            Give the output in just this format: {format_instructions}
            Output just the JSON instance and nothing else.
            Clinical note: {clinical_note}
            Keypoint: {keypoint}
            Topic: {topic}
            Context: {context}
            Question: {question}
            Correct answer: {correct_answer}
            Attempted answer: {attempted_answer}
            Reasoning: {reasoning}
            Distractor options: {distractor_options}
        """
        llm = ChatOpenAI(model=self.engine, temperature=0.7,openai_api_key=OPENAIKEY)
        with open(rubrics_path) as f:
            dictt = json.load(f)
        
        fb_parser = self.fb_dict[comp_name]
        fb_format_instructions = fb_parser.get_format_instructions()
        rub_comp_fb = {k:dictt[k] for k in dictt.keys() if k.lower().find(comp_name) != -1}
        rub_comp_fb[list(rub_comp_fb.keys())[0]] = self.setup_feedback(rub_comp_fb[list(rub_comp_fb.keys())[0]])
        fb_generator = PromptTemplate(
            template=PROMPT,
            input_variables=["component_name","clinical_note", "topic","keypoint","context","question","correct_answer","attempted_answer","reasoning","distractor_options","reasoning_rubrics"],
            partial_variables={"format_instructions": fb_format_instructions})
        fb_chain = LLMChain(llm=llm, prompt=fb_generator)
        fb_output = fb_chain.run({
            "component_name":comp_name,
            "clinical_note" : clinical_note,
            "keypoint" : keypoint,
            "topic" : topic,
            "context" : context,
            "question" : question,
            "correct_answer" : correct_answer,
            "distractor_options" : distractor_options,
            "attempted_answer":attempted_answer,
            "reasoning":reasoning,
            "reasoning_rubrics": list(rub_comp_fb.values())[0]            
        })
        print(fb_output)
        return fb_output
    def __call__(self, clinical_note: str, keypoint:str, topic:str, context:str, question:str, correct_answer:str, distractor_options:str,attempted_answer:str,reasoning:str):
        # prompt = self.make_query(clinical_note, keypoint, topic, context, question, correct_answer, distractor_options,attempted_answer,reasoning)
        #print(prompt)
        c,q,ca,do,r = self.feedback_step_by_step(self.examples_path,self.rubrics_path, clinical_note, keypoint, topic, context, question, correct_answer, distractor_options,attempted_answer,reasoning)
        c_fb = self.setup_feedback(c['context_feedback'])
        c_score = c['context_score']
        q_fb = self.setup_feedback(q['question_feedback'])
        q_score = q['question_score']
        r_fb = self.setup_feedback(r['reasoning_feedback'])
        r_score = r['reasoning_score']
        ca_fb = self.setup_feedback(ca['correct_answer_feedback'])
        ca_score = ca['correct_answer_score']
        do_fb = self.setup_feedback(do['distractor_options_feedback'])
        do_score = do['distractor_options_score']
        return c_fb,c_score,q_fb,q_score,r_fb,r_score,ca_fb,ca_score,do_fb,do_score



# if __name__ == "__main__":
    # task_feedback = UsmleQgenFeedbackLgc(
    #     prompt_examples="data/prompt/usmle/feedback.jsonl",
    #     engine="davinci-code-002"
    # )
    # test_dict =  {"clinical_note":"An 84-year-old female with a past medical history of hypertension presented with weakness, dry cough, and shortness of breath for four days. The patient had received two doses of the COVID vaccine, with the second dose in March 2021. In the ER, her vital signs were BP 133\/93, HR 103 bpm, RR 22 breaths\/min, oxygen saturation of 96% on 40 L per minute of supplemental oxygen via high-flow nasal cannula, and afebrile. Laboratory assessment is in Table. Nasopharyngeal swab for SARS-CoV-2 RNA was positive. Chest X-ray on admission shows worsening right pleural effusion with new opacity obscuring the lower two-third of the right lung and a new pleural-based opacity in the left upper lobe Figure. CT chest with contrast shows large right pleural effusion and associated right basilar consolidation and abdominal ascites. The patient was admitted to the telemetry unit and started on methylprednisolone, piperacillin-tazobactam, remdesivir, and baricitinib. The patient clinically deteriorated on Day 2 and was transferred to the intensive care unit for thoracentesis and possible intubation. Thoracentesis removed 1.95 L of bloody, serosanguineous fluid obtained, with partial resolution of the effusion Figure. On Day 3, the patient developed septic shock, florid renal failure, and lethargy and was started on intravenous fluids and norepinephrine drip. Chest X-ray showed near-complete opacification of bilateral lung fields and subsequently was intubated. On Day 4, tense ascites were noted and the patient underwent paracentesis, which removed 4.25 L of bloody, serosanguinous fluid. Renal replacement therapy started. The patient was deemed to have a guarded prognosis with multiorgan failure.","topic":"current\/short-term management","keypoint":"sepsis","content_to_fb":[{"context":"An 84-year-old female with a history of hypertension, who received two doses of the COVID-19 vaccine, presents with weakness, dry cough, and shortness of breath. She tested positive for SARS-CoV-2 RNA. Despite therapeutic interventions including methylprednisolone, piperacillin-tazobactam, remdesivir, and baricitinib, her clinical condition deteriorated. She was moved to the ICU, where she developed septic shock, florid renal failure, and lethargy, necessitating intravenous fluids and a norepinephrine drip. Her chest X-ray showed near-complete opacification of bilateral lung fields, and she was subsequently intubated. Tense ascites were noted, and after paracentesis, renal replacement therapy was initiated. The patient's prognosis is guarded due to multiorgan failure.","question":"In the context of sepsis management, what would be the next most appropriate step in the management of this patient?","topic":"current\/short-term management","keypoint":"sepsis","attempted_answer":"Correct answer: Broaden antibiotic coverage to include possible resistant organisms","reasoning":"The patient's clinical presentation indicates severe COVID-19 with multiorgan failure, septic shock, and acute respiratory distress syndrome (ARDS). Despite the therapeutic interventions including antibiotics (piperacillin-tazobactam), antiviral (remdesivir), corticosteroid (methylprednisolone), and immunomodulator (baricitinib), the patient's condition continues to deteriorate. This suggests a possible superimposed bacterial infection that may be resistant to the current antibiotic regimen. Therefore, the next most appropriate step in the management of this patient would be to broaden the antibiotic coverage to include possible resistant organisms (Option A). \n\nOptions B (discontinue norepinephrine) and D (discontinue renal replacement therapy) are inappropriate as the patient is in septic shock and renal failure. Both these therapies are crucial in supporting the patient's hemodynamic status and kidney function. Option C (administer a second dose of the COVID-19 vaccine) is not relevant in the acute management of a critically ill patient with active COVID-19 infection. Finally, option D (initiate physical therapy) would not address the immediate life-threatening issues the patient is facing.","correct_answer":"broaden antibiotic coverage to include possible resistant organisms","distractor_options":"a : discontinue norepinephrine b : administer a second dose of the covid-19 vaccine c : initiate physical therapy d : discontinue renal replacement therapy","context_feedback":"relevance:The context is highly relevant to the question and focuses on the key clinical information related to sepsis. Score: 5\/5.\nconcision:The context is concise and clearly highlights the key clinical information. Score: 5\/5.\ncoherent:The context is very coherent and flows well into the question. Score: 5\/5.\nconsistent:The context is consistent with the information in the clinical note and the topic. Score: 5\/5.\nspecific:The context is specific to the keypoint of sepsis. Score: 5\/5.\nfluent:The context is well-written and the flow of ideas is good. Score: 5\/5.\nclueing:The context does not give away the answer and maintains a good level of difficulty. Score: 5\/5.\noccurrence:The correct answer, broadening antibiotic coverage, is not evident in the context, making the question more challenging. Score: 5\/5.\ncompleteness:The context is complete with all necessary clinical information required to answer the question. Score: 5\/5.\nmisdirection:The context avoids misleading the test taker, which is good. Score: 5\/5.","context_score":"50\/50","question_feedback":"relevance:The question is highly relevant and answerable from the context. Score: 5\/5.\nclear:The question is clear and well formulated. Score: 5\/5.\nconcluding:The flow of ideas from the context to the question is very organic and seamless. Score: 5\/5.\ndifficulty:The question is appropriately challenging as the correct answer is not evident in the context. Score: 5\/5.\nclarity:The question is clear and without ambiguity. Score: 5\/5.","question_score":"25\/25","correct_answer_feedback":"relevance:The correct answer is the keypoint, which is appropriate. Score: 5\/5.\noccurrence:The correct answer, broadening antibiotic coverage, is not evident in the context, making the question more challenging. Score: 5\/5.\njustification:The correct answer is logically supported by the context and aligns well with the provided information. Score: 5\/5.\ndepth_of_understanding:The correct answer demands a nuanced understanding of sepsis management, which is good. Score: 5\/5.\nprevention_of_guesswork:The correct answer aligns with the context and avoids common misconceptions. Score: 5\/5.","correct_answer_score":"25\/25","distractor_option_feedback":"format:The distractors are formatted well and consistent. Score: 5\/5.\nlength:The distractors' lengths are similar to the correct answer. Score: 5\/5.\nrelation:The distractors are related to the correct answer, which enhances the question's quality. Score: 5\/5.\nvariation:The distractors are distinct from each other and from the correct answer. Score: 5\/5.\nplausibility:The distractors are plausible and challenge critical thinking. Score: 5\/5.\ndifferentiation:The distractors are distinct, and the correct answer clearly outshines them based on the context. Score: 5\/5.\ncommon_mistakes:The distractors align with common misconceptions, which is good. Score: 5\/5.","distractor_option_score":"35\/35","reasoning_feedback":"correctness:The attempted answer is the same as the correct answer, which is good. Score: 5\/5.\nlogical_flow:The reasoning exhibits a coherent sequence of steps that are easy to follow. Score: 5\/5.\nevidence_based_reasoning:The answer is supported by evidence from the context, which is good. Score: 5\/5.\nconsideration_of_options:The reasoning demonstrates critical evaluation of each option, which is excellent. Score: 5\/5.","reasoning_score":"20\/20"}],"status":"success"}
    
    # task_feedback.__call__(clinical_note =  test_dict['clinical_note'], 
    #             keypoint=  test_dict['keypoint'], 
    #             topic=  test_dict['topic'], 
    #             context=  test_dict["content_to_fb"][0]['context'], 
    #             question=  test_dict["content_to_fb"][0]['question'], 
    #             correct_answer=  test_dict["content_to_fb"][0]['correct_answer'], 
    #             distractor_options=  test_dict["content_to_fb"][0]['distractor_options'],
    #             attempted_answer=  test_dict["content_to_fb"][0]['attempted_answer'],
    #             reasoning=  test_dict["content_to_fb"][0]['reasoning'],
    #             examples_path = "data/prompt/usmle/feedback.jsonl",
    #             rubrics_path = "data/prompt/usmle/reasoning_rubrics.jsonl")

    # print(task_feedback.prompt)
    
 