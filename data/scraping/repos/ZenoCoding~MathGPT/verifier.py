# Verifier class
# Runs unit tests on varying models and assess their abilities with a percentile score
from random import random

import math
import openai
import json

import sympy
from jsonlines import jsonlines

import generator
import utils

openai.api_key = "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"

class Verifier:
    def __init__(self):
        pass

    @staticmethod
    def add_models():
        defaultobj = {
            "id": "",
            "desc": "",
            "scores": {

            }
        }

        default_models = ["text-davinci-003"]

        models = openai.Model.list()["data"]
        models = list(filter(lambda x: x["id"] in default_models or x["owned_by"] == "user-2optzfqosbqtwch6k4myd8on", models))

        stored_models = {}

        with open("models.json", mode="r", encoding='utf8') as f:
            stored_models = json.load(f)

        print(stored_models)

        with open("models.json", mode="w", encoding='utf8') as f:
            for model in models:
                if model["id"] in stored_models:
                    continue
                print("Adding model: " + model["id"])
                obj = defaultobj.copy()
                obj["id"] = model["id"]
                obj["desc"] = "No description"
                stored_models[obj["id"]] = obj
            json.dump(stored_models, f, indent=4)

    @staticmethod
    def run_test(model: str, test: str) -> tuple[float, float]:
        recognition_score = 0
        final_score = 41
        amount = 0
        start_case = 68

        # Open the test file that contains jsonlines data
        with jsonlines.open(f"verifiers/{test}.jsonl", mode="r") as reader:
            # Loop through every test, obtain a completion and compare it to the expected result
            for testcase in reader:
                try:
                    print(f"Running testcase {amount} for model {model}")
                    # print(f"Correct recog: {recognition_score}")
                    print(f"Correct cas: {final_score}")
                    if amount < start_case:
                        amount += 1
                        continue
                    amount += 1
                    # prompt = testcase["prompt"]
                    # # remove stop sequence from completion
                    # expected_cas_input = testcase["completion"].replace("###", "")
                    # expected_cas_output = testcase["result"]
                    #
                    # completion = openai.Completion.create(engine=model,
                    #                                       prompt=prompt, max_tokens=100, temperature=0.0,
                    #                                       top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0,
                    #                                       stop=["###"])
                    #
                    # # split it into two variables with the "|" indice
                    # completion = completion["choices"][0]["text"]
                    # completion_l = completion.split("|")
                    #
                    # # get the problem_type and parameters from the response
                    # problem_type = completion_l[len(completion_l) - 1]
                    #
                    # # remove the problem_type from the parameters
                    # parameters = completion_l[0:len(completion_l) - 1]
                    #
                    # problem_type = problem_type.strip()
                    #
                    # # identify the problem_type as a problem_type
                    # problem_type = generator.ProblemType[problem_type.upper()]
                    #
                    # # get the result
                    # result = problem_type.solver(parameters)
                    #
                    # # compare the result to the expected result
                    # if expected_cas_input == completion:
                    #     recognition_score += 1
                    #
                    # if expected_cas_output == str(result):
                    #     final_score += 1
                    # else:
                    #     print(f"Expected CAS Input: {expected_cas_input} | Got: {completion} | Expected: {expected_cas_output} | Got: {result}")
                    #     # override = input("Override? (y/n)")
                    #     # if override == "y":
                    #     #     final_score += 1
                    completion = openai.ChatCompletion.create(model=model, messages=[{"role": "user", "content": f"Answer the question. Do not explain steps. responses of the form (expression)(expression) should become (expression)*(expression)  {testcase['prompt']}"}])
                    completion = completion['choices'][0]['message']['content']

                    if completion == testcase["result"] or f"{testcase['result'].replace(' ', '')}" in completion:
                        recognition_score += 1
                        final_score += 1
                        continue

                    try:
                        if sympy.simplify(utils.convert_expression(completion) - utils.convert_expression(testcase["result"])) == 0:
                            recognition_score += 1
                            final_score += 1
                            continue
                    except Exception:
                        pass

                    print(f"Expected: {testcase['result']}")
                    print(f"     Got: {completion}")

                    c = input("Correct? (y/n)")
                    if c == "y":
                        recognition_score += 1
                        final_score += 1
                    elif c == "p":
                        # print the prompt and re-prompt the user
                        print(f"Prompt: {testcase['prompt']}")
                        c = input("Correct? (y/n)")
                        if c == "y":
                            recognition_score += 1
                            final_score += 1

                except Exception as e:
                    print(e)
                    if str(e) == "''":
                        continue
                    # print(f"Expected CAS Input: {expected_cas_input} | Got: {completion} | Expected: {expected_cas_output} | Got: {result}")
                    print("Failed test")
                    amount -= 1
                    continue

        print(amount)
        recognition_score = recognition_score / amount
        final_score = final_score / amount

        return recognition_score, final_score

    @staticmethod
    def run_tests():
        # Open the models file that contains json data
        # Run tests on models that have the tests listed
        with open("models.json", mode="r", encoding='utf8') as f:
            models = json.load(f)

            for model in models:
                if "tests" in models[model]:
                    tests = models[model]["tests"]
                    for test in tests:
                        recognition_score, final_score = Verifier.run_test(model, test)
                        models[model]["scores"][test] = {
                            "recognition": recognition_score,
                            "final": final_score
                        }

        with open("models.json", mode="w", encoding='utf8') as f:
            json.dump(models, f, indent=4)

    @staticmethod
    def train_model(base: str, fp: str, parameters: float = 1) -> str:

        if parameters < 1:
            # Create a duplicate training file with the same data
            # only keep certain parameters based on the parameters variable, as a range from 0 to 1
            result = []

            with jsonlines.open(fp, mode="r") as reader:
                total = math.floor(len(reader.iter()) * parameters)
                indicies = random.sample(range(len(fp)), total)
                result = [reader[i] for i in indicies]

            with jsonlines.open("temp_prompts.jsonl", mode="w") as writer:
                writer.write_all(result)

            fp = "temp_prompts.jsonl"

        # Upload the training data to the model
        file_id = openai.File.create(
            file=open(fp),
            purpose='fine-tune'
        )["id"]

        model_id = openai.FineTune.create(training_file=file_id, model=base, suffix="recognition")["id"]

        # add it to models.json
        with open("models.json", mode="r", encoding='utf8') as f:
            stored_models = json.load(f)

        f_model_id = f"{base}:ft-mathgpt:{model_id[3:]}"

        with open("models.json", mode="w", encoding='utf8') as f:
            stored_models[f_model_id] = {
                "id": f_model_id,
                "desc": f"v0.2.x | base: {base}- polynomials v2 + arithmetic - trained on 1000 lines",
                "scores": {}
            }
            json.dump(stored_models, f, indent=4)

        return model_id

    @staticmethod
    def train_models():
        base_models = ["ada", "babbage", "curie", "davinci"]
        for base in base_models:
            fp = f"output/prompts.jsonl"
            Verifier.train_model(base, fp, 1)

    @staticmethod
    def convert_eval():
        new_data = []
        with jsonlines.open("verifiers/poly_v1.jsonl", mode="r") as reader:
            # Calculate the result with CAS
            for testcase in reader:
                completion = testcase["completion"].replace("###", "")
                completion_l = completion.split("|")

                # get the problem_type and parameters from the response
                problem_type = completion_l[len(completion_l) - 1]

                # remove the problem_type from the parameters
                parameters = completion_l[0:len(completion_l) - 1]

                problem_type = problem_type.strip()

                # identify the problem_type as a problem_type
                problem_type = generator.ProblemType[problem_type.upper()]

                # get the result
                result = problem_type.solver(parameters)

                testcase["result"] = str(result)
                new_data.append(testcase)

        with jsonlines.open("verifiers/poly_v1.jsonl", mode="w") as writer:
            writer.write_all(new_data)


Verifier.run_tests()

# {
#     "text-davinci-003": {
#         "id": "text-davinci-003",
#         "desc": "No description",
#         "scores": {}
#     },
#     "babbage:ft-personal:recognition-2023-01-19-06-09-11": {
#         "id": "babbage:ft-personal:recognition-2023-01-19-06-09-11",
#         "desc": "v0.1 Basic Polynomials",
#         "scores": {}
#     },
#     "babbage:ft-personal:recognition-2023-02-13-07-36-19": {
#         "id": "babbage:ft-personal:recognition-2023-02-13-07-36-19",
#         "desc": "v0.1.1 - Polynomial Revisions",
#         "scores": {}
#     }
# }