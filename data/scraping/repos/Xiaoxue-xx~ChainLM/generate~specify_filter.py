import argparse
import openai
import time
import json

openai.api_key = 'sk-'

def get_res_batch(question, cot1, cot2):
    message = [
        {"role": "user", "content": "Given a question and two chain-of-thought answers to the question, try your best to judge whether #CoT 2# is better than #CoT 1#. If #CoT 2# is better than #CoT 1#, write 'Yes'. If #CoT 1# is better than #CoT 2#, write 'No'.\n\n#Question#: " + question + "\n#CoT 1#: " + cot1 + "\n#CoT 2#: " + cot2 + "\n#Your Judgement#:"}
    ]
    while True:
        try:
            res = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=message,
                temperature=0.0,
                max_tokens=16
            )
            break
        except openai.error.RateLimitError:
            print('openai.error.RateLimitError\nRetrying...')
            time.sleep(20)
        except openai.error.ServiceUnavailableError:
            print('openai.error.ServiceUnavailableError\nRetrying...')
            time.sleep(20)
        except openai.error.Timeout:
            print('openai.error.Timeout\nRetrying...')
            time.sleep(20)
        except openai.error.APIError:
            print('openai.error.APIError\nRetrying...')
            time.sleep(20)
        except openai.error.APIConnectionError:
            print('openai.error.APIConnectionError\nRetrying...')
            time.sleep(20)
    
    
    print(res['choices'][0]['message']['content'])
    return res['choices'][0]['message']['content']


def get_dataset(args):
    with open(args.file, 'r', encoding="utf-8") as f:
        data = []
        for line in f:
            data.append(json.loads(line))

        i=0
        for i in range(len(data)):
            cot1 = data[i]["answer"]
            cot2 = data[i]["rewritten_cot"]
            question = data[i]["question"]

            ans = get_res_batch(question, cot1, cot2)

            if ans == "Yes" or ans == "yes":
                judgement = "Yes"
            else:
                judgement = "No"
            gen = {"id":data[i]["id"],"question": data[i]["question"], "answer": data[i]["answer"], "rewritten_cot": data[i]["rewritten_cot"], "judgement": judgement}
            dump_jsonl(gen, args.save_path, append=True)
            
            print(data[i]["id"], "completed!")
            i = i+1

def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, 'a+', encoding='utf-8') as f:
            json_record = json.dumps(data, ensure_ascii=False)
            f.write(json_record + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='')
    parser.add_argument('--save_path', type=str, default='')
    args = parser.parse_args()
    get_dataset(args)
