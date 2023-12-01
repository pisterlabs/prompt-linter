import re
from typing import Set, List
import pandas as pd
from src.prompt_lib.prompt_lib.backends import openai_api
import json
from src.utils import Prompt


class UsmleQgenFeedback(Prompt):
    def __init__(self, engine: str, prompt_examples: str,reasoning_rubrics:str, max_tokens: int = 2048) -> None:
        super().__init__(
            question_prefix="",
            answer_prefix="",
            intra_example_sep="\n\n",
            inter_example_sep="\n\n###\n\n",
        )
        self.engine = engine
        self.max_tokens = max_tokens
        self.setup_prompt_from_examples_file(prompt_examples,reasoning_rubrics)

    def setup_prompt_from_examples_file(self, examples_path: str,rubrics_path : str) -> str:
        template = """Clinical Note: {clinical_note}

Topic: {topic}

Keypoint: {keypoint}

Context: {context}

Question: {question}

Correct answer: {correct_answer}

Distractor options: {distractor_options}

Feedback and scores:

Context feedback: {context_feedback}

Context score: {context_score}

Question feedback: {question_feedback}

Question score: {question_score}

Correct answer feedback: {correct_answer_feedback}

Correct answer score: {correct_answer_score}

Distractor options feedback: {distractor_option_feedback}

Distractor options score: {distractor_option_score}"""

        reasoning_rubric_template = """Context rubrics: {context_score_rubrics}

Reasoning rubrics: {reasoning_score_rubrics}

Question rubrics: {question_score_rubrics}

Correct answer rubrics: {correct_answer_score_rubrics}

Distractor options rubrics: {distractor_option_score_rubrics}"""
        #examples_df = pd.read_json(examples_path, orient="records", lines=True)
        with open(examples_path) as f:
            dictt = json.load(f)
        examples_df = pd.DataFrame.from_dict(dictt, orient="index").T
        prompt = []
        for _, row in examples_df.iterrows():
            prompt.append(
                template.format(
                    clinical_note=row["clinical_note"],
                    topic=row["topic"],
                    keypoint=row["keypoint"],
                    context=row["context"],
                    question=row["question"],
                    correct_answer=row["correct_answer"],
                    distractor_options=row["distractor_options"],
                    context_feedback= self.setup_feedback(row['context_feedback']),
                    context_score=row["context_score"],
                    question_feedback=self.setup_feedback(row['question_feedback']),
                    question_score=row["question_score"],
                    correct_answer_feedback = self.setup_feedback(row['correct_answer_feedback']),
                    correct_answer_score=row["correct_answer_score"],
                    distractor_option_feedback = self.setup_feedback(row['distractor_options_feedback']),
                    distractor_option_score=row["distractor_options_score"],
                )
            )
        with open(rubrics_path) as f:
            rubrics_dictt = json.load(f)
        # rubrics_df = pd.DataFrame.from_dict(rubrics_dictt, orient="index").T
        # print(rubrics_df)
        rubric_prompt = []
        # for _, row in rubrics_df.iterrows():
        rubric_prompt.append(
            reasoning_rubric_template.format(
                context_score_rubrics= self.setup_feedback(rubrics_dictt['context_score_rubrics']),\
                reasoning_score_rubrics=self.setup_feedback(rubrics_dictt['reasoning_score_rubrics']),\
                question_score_rubrics = self.setup_feedback(rubrics_dictt['question_score_rubrics']),\
                correct_answer_score_rubrics = self.setup_feedback(rubrics_dictt['correct_answer_score_rubrics']),
                distractor_option_score_rubrics = self.setup_feedback(rubrics_dictt['distractor_options_score_rubrics']),
            )
        )
        print('\n\n'.join(rubric_prompt))
        instruction = """
        In addition to the scoring rubrics in the examples above,give feedback and score the context, question, correct answer,distractors and the reasoning feedback using the attempted answer(correct/incorrect) reasoning-based rubrics and their definitions below.
        Please include both the previous scoring rubrics and the following rubrics before giving the feedback for a particular topic and add up the scores for all the aspects for the total scores of each component. 
        That means the rubrics for distractor options will be: format, length, relation(from the above examples) and variation,
        plausibility, differentiation and common_mistakes from the following reasoning-based metrics, and same for the other feedback topics which are context(relevance,concision,coherent,consistent,specific,fluent,clueing,occurrence and completeness,misdirection, 10 metrics in total), 
        question(relevance,clear,concluding,difficulty and clarity, 5 metrics in total), correct answer(relevance, occurrence and justification,depth_of_understanding,prevention_of_guesswork, 5 metric in total) and reasoning feedback.
        Many of these feedback points for all components depend upon the reasoning and the attempted answer correctness so consider that while giving feedback for the context, question, correct answer and distractor options.
        """
        self.prompt =  self.inter_example_sep.join(prompt) + instruction.replace('\n','') +self.intra_example_sep+ '\n\n'.join(rubric_prompt)
        #self.prompt = self.inter_example_sep.join(prompt) + self.inter_example_sep
        #print(self.prompt)
    def setup_feedback(self,feedback):
        fb = ''
        for key in feedback.keys():
            fb += '\n' + key + ':' + feedback[key] 
        return fb
    def __call__(self, clinical_note: str, keypoint:str, topic:str, context:str, question:str, correct_answer:str, distractor_options:str,attempted_answer:str,reasoning:str):
        prompt = self.make_query(clinical_note, keypoint, topic, context, question, correct_answer, distractor_options,attempted_answer,reasoning)
        #print(prompt)
        output = openai_api.OpenaiAPIWrapper.call(
            prompt=prompt,
            engine=self.engine,
            max_tokens=self.max_tokens,
            stop_token="###",
            temperature=0.7,
        )
        
        generated_feedback = openai_api.OpenaiAPIWrapper.get_first_response(output)
        print(generated_feedback)
        context_feedback = re.search(r"Context feedback:(.*?)(?=\n\n)", generated_feedback, re.DOTALL).group(1)
        context_score = re.search(r"Context score:(.*?)(?=\n\n)", generated_feedback, re.DOTALL).group(1)
        question_feedback = re.search(r"Question feedback:(.*?)(?=\n\n)", generated_feedback, re.DOTALL).group(1)
        question_score = re.search(r"Question score:(.*?)(?=\n\n)", generated_feedback, re.DOTALL).group(1)
        reasoning_feedback = re.search(r"Reasoning feedback:(.*?)(?=\n\n)", generated_feedback, re.DOTALL).group(1)
        reasoning_score = re.search(r"Reasoning score:(.*?)(?=\n\n)", generated_feedback, re.DOTALL).group(1)
        correct_answer_feedback = re.search(r"Correct answer feedback:(.*?)(?=\n\n)", generated_feedback, re.DOTALL).group(1)
        correct_answer_score = re.search(r"Correct answer score:(.*?)(?=\n\n)", generated_feedback, re.DOTALL).group(1)
        distractor_option_feedback = re.search(r"Distractor options feedback:(.*?)(?=\n\n)", generated_feedback, re.DOTALL).group(1)
        distractor_option_score= re.search(r"Distractor options score:(.*)", generated_feedback, re.DOTALL).group(1).strip()
        distractor_option_score = distractor_option_score if len(distractor_option_score)<=5 else re.search(r"Distractor options score:(.*?)(?=\n\n)", generated_feedback, re.DOTALL).group(1)
        return context_feedback.strip(), context_score.strip(), question_feedback.strip(), question_score.strip(), reasoning_feedback.strip(), reasoning_score.strip(),correct_answer_feedback.strip(), correct_answer_score.strip(), distractor_option_feedback.strip(), distractor_option_score.strip()

    def make_query(self, clinical_note: str, keypoint:str, topic:str, context:str, question:str, correct_answer:str, distractor_options:str,attempted_answer:str,reasoning:str):
        
        question = f"""Clinical Note: {clinical_note}

                Topic: {topic}

                Keypoint: {keypoint}

                Context: {context}

                Question: {question}

                Attempted answer: {attempted_answer}

                Correct answer: {correct_answer}

                Distractor options: {distractor_options}

                Reasoning: {reasoning}
                Feedback and scores: """
        return f"""{self.prompt}{question}"""


if __name__ == "__main__":
    task_feedback = UsmleQgenFeedback(
        prompt_examples="data/prompt/usmle/feedback.jsonl",
        engine="davinci-code-002"
    )
    
    print(task_feedback.prompt)
    
 