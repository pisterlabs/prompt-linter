import os
import openai
import time
import json
from scipy.stats import somersd


# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")

instruction = \
"Does the generated response answer the question in a well-justified manner? Please generate your own response for the question first, then conduct a qualitative analysis on the generated response by taking into account your own response, and finally give me an overall quality score in [1, 2, 3, 4, 5] (1=incomprehensible and wrong, 5=clear and correct) for the given generated response by taking into account both your own response and the qualitative analysis. Note that you need to take into account both the explanation and the answer in the generated response."

example_premise = "Do i need to go for a legal divorce ? I wanted to marry a woman but she is not in the same religion , so i am not concern of the marriage inside church . I will do the marriage registered with the girl who i am going to get married . But legally will there be any complication , like if the other woman comes back one day , will the girl who i am going to get married now will be in trouble or Is there any complication ? Why is this person asking about divorce ?"

example_hypothesis = "He wants to get married to a different person ."

generic_question = "Is the Claim supported by the Situation?"

example_generated_response = "He wants to get married to a different person means he is not married to the person he wants to get married to. The answer is Yes."

example_representation = {"own response": "Yes, the Situation suggests that the person wants to get married to a different woman than the one he may have been previously involved with.",
                          "qualitative analysis": "The generated response made an assumption that \"He wants to get married to a different person means he is not married to the person he wants to get married to.\" While this may be true, it's not directly stated in the Situation. The Situation does not provide information about the current marital status of the person in question. However, the underlying premise that he wants to get married to a different person is accurate based on the context. The generated response does conclude correctly that the answer is \"Yes,\" meaning the Claim is supported by the Situation.",
                          'overall quality': 3}

example_representation = json.dumps(example_representation)

evaluation_prompt = "Please generate your own response for the question first, then conduct a qualitative analysis on the generated response by taking into account your own response, and finally give me the overall quality of the given generated response for the question by taking into account both your own response and the qualitative analysis based on the instruction and the format of the example representation:"


def is_json(myjson):
    try:
        json.loads(myjson)
    except ValueError as e:
        return False
    return True


def run_SocREval_cosmos(test_path, output_path, input_option, output_option):
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


def evaluate_SocREval_cosmos(test_path, representation_option):
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
    cosmos_roscoe_path = dir_path + 'ROSCOE/cosmos_roscoe_annotations.json'
    cosmos_roscoe_SocREval_path = dir_path + 'ROSCOE/cosmos_roscoe_annotations_SocREval.json'
    time_start = time.time()
    run_SocREval_cosmos(cosmos_roscoe_path, cosmos_roscoe_SocREval_path, input_option='gpt-3',
                        output_option='gpt-3_SocREval')
    evaluate_SocREval_cosmos(cosmos_roscoe_SocREval_path, representation_option='gpt-3_SocREval')
    time_end = time.time()
    print('\ntime:', time_end - time_start)

