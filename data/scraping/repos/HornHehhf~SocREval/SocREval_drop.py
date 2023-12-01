import os
import openai
import time
import json
from scipy.stats import somersd


# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")

instruction = \
"Does the generated response answer the question in a well-justified manner? Please generate your own response for the question first, then conduct a qualitative analysis on the generated response by taking into account your own response, and finally give me an overall quality score in [1, 2, 3, 4, 5] (1=incomprehensible and wrong, 5=clear and correct) for the given generated response by taking into account both your own response and the qualitative analysis. Note that you need to take into account both the explanation and the answer in the generated response."

example_premise = " Hoping to rebound from their loss to the Patriots, the Raiders stayed at home for a Week 16 duel with the Houston Texans.  Oakland would get the early lead in the first quarter as quarterback JaMarcus Russell completed a 20-yard touchdown pass to rookie wide receiver Chaz Schilens.  The Texans would respond with fullback Vonta Leach getting a 1-yard touchdown run, yet the Raiders would answer with kicker Sebastian Janikowski getting a 33-yard and a 30-yard field goal.  Houston would tie the game in the second quarter with kicker Kris Brown getting a 53-yard and a 24-yard field goal. Oakland would take the lead in the third quarter with wide receiver Johnnie Lee Higgins catching a 29-yard touchdown pass from Russell, followed up by an 80-yard punt return for a touchdown.  The Texans tried to rally in the fourth quarter as Brown nailed a 40-yard field goal, yet the Raiders' defense would shut down any possible attempt. Who scored the first touchdown of the game?"

example_hypothesis = "Chaz Schilens"

generic_question = "Is the Claim supported by the Situation?"

example_generated_response = "Chaz Schilens was the receiver of the touchdown pass from JaMarcus Russell, so he scored the first touchdown. The answer is Yes."

example_representation = {"own response": "Yes, the Claim is supported by the Situation as Chaz Schilens caught a 20-yard touchdown pass from JaMarcus Russell, making him the scorer of the first touchdown.",
                          "qualitative analysis": "Both the generated response and my own response acknowledge that Chaz Schilens caught the first touchdown of the game from a pass by JaMarcus Russell. The generated response provides a concise and clear explanation that supports the answer.",
                          'overall quality': 5}

example_representation = json.dumps(example_representation)

evaluation_prompt = "Please generate your own response for the question first, then conduct a qualitative analysis on the generated response by taking into account your own response, and finally give me the overall quality of the given generated response for the question by taking into account both your own response and the qualitative analysis based on the instruction and the format of the example representation:"


def is_json(myjson):
    try:
        json.loads(myjson)
    except ValueError as e:
        return False
    return True


def run_SocREval_drop(test_path, output_path, input_option, output_option):
    max_num_per_question = 5
    test_file = open(test_path, 'r')
    test_data = json.load(test_file)
    for idx, test_case in enumerate(test_data):
        print(idx, len(test_data))
        premise = test_case["premise"]
        hypothesis = test_case["hypothesis"]
        generated_response = test_case[input_option]
        combined_prompt = "Instruction:\n" + instruction \
                          + "\n\nExample Situation (Premise):\n" + example_premise \
                          + "\n\nExample Claim (Hypothesis):\n" + example_hypothesis \
                          + "\n\nExample question:\n" + generic_question \
                          + "\n\nExample generated response:\n" + example_generated_response \
                          + "\n\nExample representation:\n" + example_representation \
                          + "\n\nSituation (Premise):\n" + premise \
                          + "\n\nClaim (Hypothesis):\n" + hypothesis \
                          + "\n\nQuestion:\n" + generic_question \
                          + "\n\nGenerated response:\n" + generated_response \
                          + "\n\n" + evaluation_prompt + "\n"
        print(combined_prompt)
        for i in range(max_num_per_question):
            response = openai.ChatCompletion.create(model="gpt-4-0613",
                                                    messages=[{"role": "user", "content": combined_prompt}])
            text = response['choices'][0]['message']['content']
            if i == max_num_per_question - 1:
                test_case[output_option] = text
            if is_json(text):
                representation = json.loads(text)
                print(representation)
                test_case[output_option] = json.dumps(representation)
                break
            else:
                print('format error')
        with open(output_path, 'w') as f:
            json.dump(test_data, f, indent=4)


def evaluate_SocREval_drop(test_path, representation_option):
    test_file = open(test_path, 'r')
    test_data = json.load(test_file)
    correctness_list = []
    human_eval_score_list = []
    for idx, test_case in enumerate(test_data):
        representation = json.loads(test_case[representation_option])
        correctness = (float(representation['overall quality']) - 1) / 4
        human_eval_score = (float(test_case['0_full_newOverall_result']) - 1) / 4
        human_eval_score_list.append(human_eval_score)
        correctness_list.append(correctness)
    print('\ncorrectness somersd:', somersd(human_eval_score_list, correctness_list))


if __name__ == '__main__':
    dir_path = '/path/to/working/dir/'
    drop_roscoe_path = dir_path + 'ROSCOE/drop_roscoe_annotations.json'
    drop_roscoe_SocREval_path = dir_path + 'ROSCOE/drop_roscoe_annotations_SocREval.json'
    time_start = time.time()
    run_SocREval_drop(drop_roscoe_path, drop_roscoe_SocREval_path, input_option='gpt-3',
                      output_option='gpt-3_SocREval')
    evaluate_SocREval_drop(drop_roscoe_SocREval_path, representation_option='gpt-3_SocREval')
    time_end = time.time()
    print('\ntime:', time_end - time_start)

