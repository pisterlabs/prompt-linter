import openai
import argparse
import time 
import re

from dataset import get_data


def get_response(args, user_request, max_len, temp):
    ###
    responese = openai.Completion.create(
        engine = args.model_name,
        prompt = user_request,
        max_tokens = max_len,
        n = 1,
        temperature = temp
    )

    return responese

def convert_to_submit_file(api_result: list = []):
    answer_start = api_result.find("Answer: ")
    if answer_start != -1:
        answer_end = api_result.find(",", answer_start)
        answer_part = api_result[answer_start + len("Answer: "):answer_end]

        if any(c.isalpha() for c in answer_part):
            answer = answer_part[0:answer_part.find(")")]  
        else:
            answer = answer_part
        return answer.lower()
    else:
        answer = api_result
        return answer.lower()
    return 'Nan'
            
                

def main(args):
    with open("openai_api_key.txt", "r") as f:
        openai.api_key = f.readline() ###
        test_examples = get_data(args.data)
        results = []
        with open('./results/results.txt', 'r') as read:
            results = read.readlines()
        curr_indx = 1
        last_indx = len(results)
        print("Last request: ", last_indx)
        with open('./results/results.txt', 'a') as f:
            for problem in test_examples:
                prompt = "Help me choose the correct answer to the following problem. Note that you only need to return the letters corresponding to the chosen answer. \nQuestion:"
                ques = problem["Problem"]
                max_len = 20
                temp = 0.2
                user_request = prompt + ques
                responese = {}
                if curr_indx > last_indx:
                    while 'id' not in responese:
                        try:
                            t1 = time.time()
                            responese = get_response(args, user_request, max_len, temp)
                            #print(user_request)
                            t2 = time.time()
                            time_request = t2 - t1
                            answer = responese.choices[0].text
                            #results.append([answer, time_request])
                        except:
                            print("Waiting...")
                            time.sleep(20)
                            continue
                    print(f"Time request for {problem['id']}: {time_request}, answer: {answer}")
                    choose = convert_to_submit_file(answer)
                    f.write(choose + '\t' + str(time_request) + '\n')
                    
                curr_indx += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
		"--model_name", type=str, 
		default="text-davinci-003",
        help= "Name to request model from openai"
	)
    parser.add_argument(
		"--data", type=str, 

		default="./data/test.json",
		help="Path to data test"
	)

    args = parser.parse_args()
    main(args)
