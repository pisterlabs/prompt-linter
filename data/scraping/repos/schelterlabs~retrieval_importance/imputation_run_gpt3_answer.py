
import openai
import json

def get_openai(text, max_tokens = 100, n = 1, temperature = 0.1):
    try:
        openai.api_key = ""
        res = openai.Completion.create(model="text-davinci-003", prompt=text, n=n, max_tokens=max_tokens, temperature = temperature)
        print(res)
        answer = [i.text.lstrip().rstrip() for i in res['choices']]
        return answer[0]
    except:
        return ["Wrong" for i in range(n)]

def run(target, split):
    file_name = "./applications/imputation/data/{dataset}/{dataset}_{split}_zeroshot.txt".format(dataset=target, split=split)
    save_name = "./applications/imputation/answers/{dataset}_{split}_gpt3.jsonl".format(dataset=target, split=split)
    with open(file_name) as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    with open(save_name, "w") as f:
        for line in lines:
            answer = get_openai(line)
            meta = {"question": line, "answer": answer}
            f.write(json.dumps(meta) + "\n")
            f.flush()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', type=str, default="buy", help='the target dataset')
    parser.add_argument('-s', type=str ,default="valid", help='the split of the dataset')

    args = parser.parse_args()

    run(args.t, args.s)