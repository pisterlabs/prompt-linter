import os
import json
import argparse
import random
import time
from functools import lru_cache

from base_prompt1 import *

from utilities1 import get_gpt3_output
from model1 import policy_network
from utilities1 import extract_prediction, normalize_answer, normalize_formula
import numpy as np
import torch
import torch.nn.functional as F
import openai

import chatgpt0

# openai.api_key = "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"
# openai.api_key = "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"
openai.api_key = "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"


def read_data(args,stage="train"):
    datas = []
    with open(os.path.join(args.data_root,f'svampformula_{stage}.jsonl')) as reader:
        idx = 0
        try:
            for line in reader:
                data = json.loads(line)
                datas.append(data)
                idx = idx + 1
        except Exception as e:
            print("error:",e)
            print("idx:",idx)
    return datas

def load_data(args):
    problems_test = read_data(args,stage=args.test_split)
    problems_train = read_data(args,stage="train")
    # 输出problems_test 和 problems_train的长度
    print(f"number of test problems: {len(problems_test)}")
    print(f"number of train problems: {len(problems_train)}")

    # 合并prolems_test和problems_train
    problems = [*problems_test,*problems_train]
    testLen=len(problems_test)
    # test problem ids
    test_pids = [item for item in range((testLen))]
    test_pids = random.sample(test_pids, args.test_number) if args.test_number > 0 else test_pids
    print(f"number of test problems: {len(test_pids)}\n")
    # pick up shot/in-context example candidates from the training set
    train_pids = [item for item in range((testLen),len(problems))]

    cand_pids = random.sample(train_pids, args.cand_number)  # random sample

    return problems, test_pids, cand_pids

@lru_cache(maxsize=10000)
def call_gpt3(prompt,algebraicPrompt,engine):
        message =  algebraicPrompt+"\n\n"+prompt
        reply = chatgpt0.call_gpt3(engine,message)
        return reply



def get_result_file(args):
    result_path = f"{args.output_root}/{args.model}"
    os.makedirs(result_path, exist_ok=True)

    result_file = "{}/{}_{}_{}_{}_seed_{}.json".format(result_path, args.label, args.test_split, args.prompt_format,
                                                       args.shot_number, args.seed)

    return result_file


def save_results(result_file, acc, correct, count, cand_pids, args, results):
    data = {}
    data['acc'] = acc
    data['correct'] = correct
    data['count'] = count
    data['cand_pids'] = cand_pids
    data['args'] = vars(args)
    data['results'] = results

    with open(result_file, 'w') as f:
        json.dump(data, f, indent=2, separators=(',', ': '))



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../data/svamp')
    parser.add_argument('--output_root', type=str, default='../results')
    parser.add_argument('--model', type=str, default='gpt3_rl_formula')
    parser.add_argument('--option_inds', type=list, default=["A", "B", "C", "D", "E", "F"])

    # user options
    parser.add_argument('--label', type=str, default='svampexp')
    parser.add_argument('--test_split', type=str, default='test', choices=['dev', 'dev1k', 'test', 'test1k'])
    parser.add_argument('--test_number', type=int, default=100, help='GPT-3 is expensive. -1 for the whole test set')
    parser.add_argument('--save_every', type=int, default=10, help='Save the result with every n examples.')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument(
        '--prompt_format',
        type=str,
        default='SVAMP',
        choices=['T-A', 'Q-A', 'Q-AS', 'Q-SA', 'TQ-A', 'TQ-AS', 'TQ-SA', 'QT-A', 'QT-AS', 'QT-SA', 'QTS-A', 'TQS-A'],
        help='prompt format template')
    parser.add_argument('--shot_number', type=int, default=2, help='Number of n-shot training examples.')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    # GPT-3 settings
    parser.add_argument('--engine', type=str, default='text-davinci-002', choices=['text-davinci-002', 'ada'])
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max_tokens',
                        type=int,
                        default=512,
                        help='The maximum number of tokens allowed for the generated answer.')
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)

    # Policy Model settings
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--model_config',
                        type=str,
                        default='bert-base-uncased',
                        )
    parser.add_argument('--cand_number', type=int, default=20, help='Number of candidate prompts.')
    parser.add_argument('--embedding_size', type=int, default=128, help='Policy network final layer hidden state size.')
    parser.add_argument('--ckpt_root', type=str, default='../checkpoints')
    parser.add_argument('--ckpt', type=str, default=None)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    # https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)  # CPU random seed
    torch.cuda.manual_seed(args.seed)  # GPU random seed
    torch.backends.cudnn.benchmark = True

    # problems, test question ids, candidate prompt pids, RL training pids
    problems, pids, cand_pids = load_data(args)

    result_file = get_result_file(args)

    # load the check point
    if os.path.exists(result_file):
        print("# The result file exists! We will load the learned check point!!!")
        check_point = json.load(open(result_file))
        results = check_point['results']
    else:
        results = {}

    total = len(pids)
    check_count = len(results)  # number of existing results
    correct = 0  # number of correct results

    # policy network
    policy_model = policy_network(model_config=args.model_config,
                                  add_linear=True,
                                  embedding_size=args.embedding_size,
                                  freeze_encoder=True)

    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")  # one GPU
    policy_model = policy_model.to(device)

    # print("candidate prompts: ")
    # print("===========")
    cand_examples = []
    for pid in cand_pids:
        testStage = True
        example = create_example_from_pid(pid, problems, testStage=True)  # CHECK !!!
        # print(example)
        # print("===========")
        cand_examples.append(example)

    # ======================================================= INFERENCE ===============================================
    if args.ckpt:
        ckpt_path = os.path.join(args.ckpt_root, args.ckpt)
        if os.path.exists(ckpt_path):
            policy_model.linear.load_state_dict(torch.load(ckpt_path))
        else:
            print(f"The ckpt path for [{ckpt_path}] does not exist!")  # CHECK
            exit()
    else:
        print(f"!!! Load the pre-traind model instead!")  # CHECK
        # exit()

    policy_model.eval()

    with torch.no_grad():

        # Calculate the embeddings for candidate examples only one time!
        cand_embedding = policy_model(cand_examples)
        # print("cand_embedding:", cand_embedding.shape)  # [cand_num x emb_size]

        for i, pid in enumerate(pids):
            count = i + 1  # number of current results
            problem = problems[pid]
            # print("problem的属性值",problem.keys())
            try:
                infixEquation = problems[pid]['infixEquation']
                # print("answer的属q性值",answer)
                answer = normalize_answer(problems[pid]['Answer'])
                example = create_example_from_pid(pid, problems, testStage=True)
            except Exception as e:
                print("there is a error",e)
                break
            ctxt_embedding = policy_model([example])
            # print("ctxt_embedding:", ctxt_embedding.shape)  # [1 x emb_size]
            scores = F.softmax(torch.mm(ctxt_embedding, cand_embedding.t()), dim=1)[0]  # [cand_num]
            # print(scores.shape)
            scores = scores.cpu().detach().numpy().tolist()
            cand_ids = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:args.shot_number]
            shot_pids = [cand_pids[cid] for cid in cand_ids[::-1]]
            prompt = build_prompt(problems, shot_pids, pid)  # generate the prompt input
            if pid in results:
                output = results[pid]["infixQuation"]
            else:
                raw_formula,formula,prediction = get_gpt3_output(prompt, args,problems[pid]["Numbers"],pid)
            # save the results
            results[pid] = {}
            results[pid]["Question"] = problems[pid]["Question"]
            results[pid]["shot_pids"] = shot_pids
            # results[pid]["prompt"] = prompt
            results[pid]["Numbers"] = problems[pid]["Numbers"]
            results[pid]["infixEquation"] = infixEquation
            results[pid]["Answer"] = answer

            results[pid]["rawEquation"] = raw_formula
            results[pid]["infixEquationPre"] = formula
            if prediction:
                results[pid]["prediction"] = prediction
            else:
                results[pid]["prediction"] = None



            # correct or not
            if answer == prediction:
                correct += 1
                results[pid]["true_false"] = True
            else:
                results[pid]["true_false"] = False

            acc = correct / (i + 1) * 100

            if args.debug or i < 10:
                print("\n##################################")
                print(prompt, "\n")
                print("[A] labeled answer (normalized):\t", answer)
                print("[P] predicted answer (normalized):\t", prediction)
                print("[Acc]:\t", results[pid]["true_false"])
                print("")
                print("[A] labeled answer:\t", answer)
                print("[P] predicted answer:\t", prediction)


            if count % args.save_every == 0 or count == total:
                if count >= check_count:
                    # have new outputs
                    print(f"{count}/{total}, correct: {correct}, acc: {round(acc, 2)}%, saved to {result_file}")
                    save_results(result_file, acc, correct, count, cand_pids, args, results)
                else:
                    # no new outputs, just print the accuracy
                    print(f"{count}/{total}, correct: {correct}, acc: {round(acc, 2)}%")
