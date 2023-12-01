# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : Answer_Method.py
@Project  : SelfPolish
@Time     : 2023/5/30 14:07
@Author   : Zhiheng Xi
"""

from openai_utils import *
from utils import *
import traceback


class AnswerMethod():
    def __init__(self,answer_method="few_shot",
                 eng="text-davinci-003",
                 few_shot_prompt_path=None
                 ):
        self.answer_method = answer_method
        self.eng = eng

        pass
    def forward(self,question,answer,direct_answer_trigger_for_fewshot,dataset):
        pass

    def generate_CoT_and_answer(self,eng,method,direct_answer_trigger_for_fewshot,dataset, prompt_input):
        if eng in ["gpt-3.5-turbo", "gpt-4"]:
            # gpt4
            CoT_and_answer = create_response_chat(
                eng,
                prompt_input=[
                    {"role": "system", "content": "Follow the given examples and answer the question."},
                    {"role": "user", "content": prompt_input}
                ],
                temperature=0,
                max_tokens=256
            )["choices"][0]["message"]["content"]
            time.sleep(21)
        else:
            CoT_and_answer = create_response(
                eng,
                prompt_input=prompt_input,
                temperature=0,

                max_tokens=256
            )["choices"][0]["text"]

        answer = answer_cleansing(method, direct_answer_trigger_for_fewshot, dataset,
                                  CoT_and_answer.split(direct_answer_trigger_for_fewshot)[-1])
        return CoT_and_answer, answer


class AnswerFewShot(AnswerMethod):
    def __init__(self,answer_method="few_shot",
                 eng="text-davinci-003",
                 few_shot_prompt_path=None
                 ):
        self.answer_method = answer_method
        self.eng = eng
        assert few_shot_prompt_path is not None
        with open(file=few_shot_prompt_path, mode="r", encoding="utf-8") as f:
            self.few_shot_prompt = f.read().strip()

    def forward(self, question,answer,direct_answer_trigger_for_fewshot,dataset):
        try:
            prompt_input_to_generate_CoT_and_answer_for_original_question = self.few_shot_prompt + "\n\nQ: {}\nA:".format(
                question)
            original_CoT_and_answer, original_answer = self.generate_CoT_and_answer(eng=self.eng,
                                                                                    method=self.answer_method,
                                                                                    direct_answer_trigger_for_fewshot=direct_answer_trigger_for_fewshot,
                                                                                    dataset=dataset,
                                                                                    prompt_input=prompt_input_to_generate_CoT_and_answer_for_original_question)
            print("Question: {}".format(question))
            print("Ans or CoT and Ans: {}".format(original_CoT_and_answer))
            print("Ans: {}".format(original_answer))
            print("golden: {}".format(answer))

            return original_answer
        except Exception as e:
            print(repr(e))
            traceback.print_exc()
            raise e

class AnswerFewShotCoT(AnswerMethod):
    def __init__(self,answer_method="few_shot_cot",
                 eng="text-davinci-003",
                 few_shot_prompt_path=None
                 ):
        self.answer_method = answer_method
        self.eng = eng
        assert few_shot_prompt_path is not None
        with open(file=few_shot_prompt_path, mode="r", encoding="utf-8") as f:
            self.few_shot_prompt = f.read().strip()

    def forward(self, question,answer,direct_answer_trigger_for_fewshot,dataset):
        try:
            prompt_input_to_generate_CoT_and_answer_for_original_question = self.few_shot_prompt + "\n\nQ: {}\nA:".format(
                question)
            original_CoT_and_answer, original_answer = self.generate_CoT_and_answer(eng=self.eng,
                                                                                    method=self.answer_method,
                                                                                    direct_answer_trigger_for_fewshot=direct_answer_trigger_for_fewshot,
                                                                                    dataset=dataset,
                                                                                    prompt_input=prompt_input_to_generate_CoT_and_answer_for_original_question)
            print("Question: {}".format(question))
            print("Ans or CoT and Ans: {}".format(original_CoT_and_answer))
            print("Ans: {}".format(original_answer))
            print("golden: {}".format(answer))

            return original_answer
        except Exception as e:
            print(repr(e))
            traceback.print_exc()
            raise e

class AnswerFewShotLtM(AnswerMethod):
    def __init__(self,answer_method="least_to_most",
                 eng="text-davinci-003",
                 problem_reducing_prompt_path=None,
                 problem_solving_prompt_path=None
                 ):
        self.answer_method = answer_method
        self.eng = eng
        assert problem_reducing_prompt_path is not None
        assert problem_solving_prompt_path is not None
        with open(file=problem_reducing_prompt_path, mode="r", encoding="utf-8") as f:
            self.problem_reducing_prompt = f.read().strip()
        with open(file=problem_solving_prompt_path, mode="r", encoding="utf-8") as f:
            self.problem_solving_prompt = f.read().strip()

    def forward(self,question,answer,direct_answer_trigger_for_fewshot,dataset):
        try:
            problem_reducing_prompt_input = self.problem_reducing_prompt + "\n\nQ: {}\nA:".format(question)
            # reduce
            if self.eng in ["gpt-3.5-turbo", "gpt-4"]:
                # gpt4
                reduced_problem = create_response_chat(
                    eng=self.eng,
                    prompt_input=[
                        {"role": "system", "content": "Follow the given examples and answer the question."},
                        {"role": "user", "content": problem_reducing_prompt_input}
                    ],
                    temperature=0,
                    max_tokens=256
                )["choices"][0]["message"]["content"]
                time.sleep(20)
            else:
                reduced_problem = create_response(
                    eng=self.eng,
                    prompt_input=problem_reducing_prompt_input,
                    temperature=0,
                    max_tokens=256
                )["choices"][0]["text"]

            print("Reducing Response: {}".format(reduced_problem))

            reduced_problem_list = reduced_problem.split("\"")
            reduced_problem_list = [i for i in reduced_problem_list if "?" in i]
            final_q = reduced_problem_list[0]
            # reduced_problem_list.reverse()
            reduced_problem_list = reduced_problem_list[1:] + [final_q]

            print("---Reduced to {} questions. They are: {}".format(len(reduced_problem_list), reduced_problem_list))

            # solve
            problem_solving_prompt_input = self.problem_solving_prompt + "\n\n{}".format(question)
            sub_answer = None

            for sub_q_count in range(len(reduced_problem_list)):
                sub_q = reduced_problem_list[sub_q_count]
                problem_solving_prompt_input = problem_solving_prompt_input + "\n\nQ: {}\nA:".format(sub_q)
                # solve
                print("-------------------------------------")
                # if sub_q_count == len(reduced_problem_list) - 1:
                #     print("Current sub prompt input: {}".format(problem_solving_prompt_input))
                print("---Current sub question {}: {}".format(sub_q_count, sub_q))
                if self.eng in ["gpt-3.5-turbo", "gpt-4"]:
                    # gpt4
                    sub_answer = create_response_chat(
                        eng=self.eng,
                        prompt_input=[
                            {"role": "system", "content": "Follow the given examples and answer the question."},
                            {"role": "user", "content": problem_solving_prompt_input}
                        ],
                        temperature=0,
                        max_tokens=256,
                        stop="Q"
                    )["choices"][0]["message"]["content"]
                    time.sleep(20)
                else:
                    sub_answer = create_response(
                        eng=self.eng,
                        prompt_input=problem_solving_prompt_input,
                        temperature=0,
                        max_tokens=256,
                        stop="Q"
                    )["choices"][0]["text"]
                problem_solving_prompt_input = problem_solving_prompt_input + sub_answer
                print("---Current sub answer: {}".format(sub_answer))

            least_to_most_answer = answer_cleansing(method=self.answer_method, direct_answer_trigger_for_fewshot=direct_answer_trigger_for_fewshot,
                                                    dataset=dataset,
                                                    pred=sub_answer.split(direct_answer_trigger_for_fewshot)[-1]
                                                    )
            # least_to_most_answer = answer_cleansing(args, "3/5")
            print("Question: {}".format(question))
            print("Ans: {}".format(least_to_most_answer))
            print("golden: {}".format(answer))
            return least_to_most_answer

        except Exception as e:
            raise e

if __name__ == '__main__':
    pass
