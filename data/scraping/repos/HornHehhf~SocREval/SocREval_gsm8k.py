import os
import openai
import time
import json
from scipy.stats import somersd

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")

instruction = \
"Does the generated response answer the question in a well-justified manner? Please generate your own response for the question first, then conduct a qualitative analysis on the generated response by taking into account your own response, and finally give me an overall quality score in [1, 2, 3, 4, 5] (1=incomprehensible and wrong, 5=clear and correct) for the given generated response by taking into account both your own response and the qualitative analysis. Note that you need to take into account both the explanation and the answer in the generated response."

example_question = \
    "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

example_generated_response = \
    "Janet eats 3 duck eggs for breakfast and bakes 4 into muffins so 3 + 4 = <<3+4=7>>7 duck eggs are used\nEach day Janet's ducks lay 16 eggs and she uses 7, 16 - 7 = <<16-7=9>>9 duck eggs are for sale\nShe sells her eggs for $2 per egg and has 9 available for sale so 2 * 9 = $<<2*9=18>>18 per day\nA: 18\n"

example_representation = {'own response': "Janet uses a total of 3 (for breakfast) + 4 (for muffins) = 7 duck eggs each day. Given that her ducks lay 16 eggs per day, she is left with 16 - 7 = 9 eggs to sell at the farmers' market. Each of these eggs sells for $2, so Janet makes a total of 9 * $2 = $18 at the farmers' market each day.",
                          "qualitative analysis": "The generated response is correct and complete. It precisely calculates the number of eggs that Janet consumes each day and subtracts that from the total number of eggs laid. It then multiplies the remaining eggs by the selling price to get the total amount that Janet makes each day. The response is also clear and easy to follow, with each step of the calculation laid out explicitly.",
                          'overall quality': 5}

example_representation = json.dumps(example_representation)

evaluation_prompt = "Please generate your own response for the question first, then conduct a qualitative analysis on the generated response by taking into account your own response, and finally give me the overall quality of the given generated response for the question by taking into account both your own response and the qualitative analysis based on the instruction and the format of the example representation:"


def is_json(myjson):
    try:
        json.loads(myjson)
    except ValueError as e:
        return False
    return True


def run_SocREval_gsm8k(test_path, output_path, input_option, output_option):
    max_num_per_question = 5
    test_file = open(test_path, 'r')
    test_data = json.load(test_file)
    for idx, test_case in enumerate(test_data):
        print(idx, len(test_data))
        question = test_case['question']
        generated_response = test_case[input_option]['solution']
        combined_prompt = "Instruction:\n" + instruction \
                          + "\n\nExample question:\n" + example_question \
                          + "\n\nExample generated response:\n" + example_generated_response \
                          + "\n\nExample representation:\n" + example_representation \
                          + "\n\nQuestion:\n" + question \
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


def evaluate_SocREval_gsm8k(test_path, representation_option):
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
    gsm8k_roscoe_path = dir_path + 'ROSCOE/gsm8k_roscoe_annotations.json'
    gsm8k_roscoe_SocREval_path = dir_path + 'ROSCOE/gsm8k_roscoe_annotations_SocREval.json'
    time_start = time.time()
    run_SocREval_gsm8k(gsm8k_roscoe_path, gsm8k_roscoe_SocREval_path, input_option='175b_verification',
                       output_option='175b_verification_SocREval')
    evaluate_SocREval_gsm8k(gsm8k_roscoe_SocREval_path, representation_option='175b_verification_SocREval')
    time_end = time.time()
    print('\ntime:', time_end - time_start)

