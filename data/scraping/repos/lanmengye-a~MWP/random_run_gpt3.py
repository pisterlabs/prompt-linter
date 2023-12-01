import os
import json
import argparse
import random
import time

from base_prompt_correct import *
from model import *
from tools.utilities1 import extract_prediction, normalize_answer

import numpy as np
import torch
import torch.nn.functional as F
import openai
from tools.utilities1 import get_gpt3_output


# openai.api_key = "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"


def load_data(args):
    problems = [json.loads(line) for line in open("dataset/svamp/svamp_test.jsonl", "r")]
    pids = [item["Index"] - 1 for item in problems]
    # samples = random.sample(pids, args.test_number)  # random sample
    # test_pids = samples[:args.test_number]
    # analysis with best run_gpt3
    with open(f"results/gpt3_rl/exp0learn_policy_autopot_selfask_epsilon99_test_TQ-A_2_best_seed_{args.seed}.json") as reader:
        lines = json.loads(reader.read())
    result = lines["results"]
    pids = result.keys()
    test_pids = [int(pid) for pid in list(pids)]
    print("test pids:",test_pids)
    input("continue?")
    return problems, test_pids


def get_result_file(args):
    result_path = f"{args.output_root}/{args.model}"
    os.makedirs(result_path, exist_ok=True)

    result_file = "{}/{}_{}_{}_{}_seed_{}.json".format(result_path, args.label, args.test_split, args.prompt_format,
                                                       args.shot_number, args.seed)

    return result_file


def save_results(result_file, acc, correct, count, args, results):
    data = {}
    data['acc'] = acc
    data['correct'] = correct
    data['count'] = count

    data['args'] = vars(args)
    data['results'] = results

    with open(result_file, 'w') as f:
        json.dump(data, f, indent=2, separators=(',', ': '))




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../data/tabmwp')
    parser.add_argument('--output_root', type=str, default='results')
    parser.add_argument('--model', type=str, default='gpt3_rl')
    parser.add_argument('--option_inds', type=list, default=["A", "B", "C", "D", "E", "F"])

    # user options
    parser.add_argument('--label', type=str, default='exp0learn_policy_autopot_selfask_epsilon99')
    parser.add_argument('--test_split', type=str, default='test', choices=['dev', 'dev1k', 'test', 'test1k'])
    parser.add_argument('--test_number', type=int, default=100, help='GPT-3 is expensive. -1 for the whole test set')
    parser.add_argument('--save_every', type=int, default=10, help='Save the result with every n examples.')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument(
        '--prompt_format',
        type=str,
        default='TQ-A',
        choices=['T-A', 'Q-A', 'Q-AS', 'Q-SA', 'TQ-A', 'TQ-AS', 'TQ-SA', 'QT-A', 'QT-AS', 'QT-SA', 'QTS-A', 'TQS-A'],
        help='prompt format template')
    parser.add_argument('--shot_number', type=int, default=2, help='Number of n-shot training examples.')
    parser.add_argument('--seed', type=int, default=999, help='random seed')

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
                        choices=['distilbert-base-uncased', 'bert-base-uncased'])
    parser.add_argument('--cand_number', type=int, default=20, help='Number of candidate prompts.')
    parser.add_argument('--embedding_size', type=int, default=128, help='Policy network final layer hidden state size.')
    parser.add_argument('--ckpt_root', type=str, default='checkpoints')
    parser.add_argument('--ckpt', type=str, default="exp0learn_policy_autopot_selfask_epsilon99/ckpt_best_reward.pt")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    from run_gpt3_rl_pot.tools import utils

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
    problems, pids = load_data(args)

    result_file = get_result_file(args)
    args.ckpt_path = os.path.join(args.ckpt_root, args.label)
    utils.create_dir(args.ckpt_path)
    logger = utils.Logger(os.path.join(args.ckpt_path, f'log_eval_random_{args.seed}.txt'))
    # load the check point
    if os.path.exists(result_file):
        logger.write("# The result file exists! We will load the learned check point!!!")
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

    # logger.write("candidate prompts: ")
    # logger.write("===========")
    cand_examples = ["Yes", "No"]

    # ======================================================= INFERENCE ===============================================
    if args.ckpt:
        ckpt_path = os.path.join(args.ckpt_root, args.ckpt)
        if os.path.exists(ckpt_path):
            policy_model.linear.load_state_dict(torch.load(ckpt_path))
        else:
            logger.write(f"The ckpt path for [{ckpt_path}] does not exist!")  # CHECK
            exit()
    else:
        logger.write(f"!!! Load the pre-traind model instead!")  # CHECK
        # exit()

    policy_model.eval()

    with torch.no_grad():

        # Calculate the embeddings for candidate examples only one time!
        cand_embedding = policy_model(cand_examples)
        # logger.write("cand_embedding:", cand_embedding.shape)  # [cand_num x emb_size]
        shottimes = 0
        for i, pid in enumerate(pids):
            count = i + 1  # number of current results
            problem = problems[pid]
            answer = problem['Answer']
            # example = create_example_from_pid(pid, problems, args, test=True)
            example = problem["Question"]

            ctxt_embedding = policy_model([example])
            # logger.write("ctxt_embedding:", ctxt_embedding.shape)  # [1 x emb_size]

            scores = F.softmax(torch.mm(ctxt_embedding, cand_embedding.t()), dim=1)[0]  # [cand_num]
            # logger.write(scores.shape)
            scores = scores.cpu().detach().numpy().tolist()
            import random
            shot_pids = random.sample(range(len(scores)),1)
            shot_pid = shot_pids[0]
            shottimes += shot_pid
            cand_examples = ["Yes", "No"]
            follow_up = cand_examples[shot_pid]
            from annotated_demos.auto_annotated.self_ask.one_prompt import MATH_PROMPT

            few_prompt = MATH_PROMPT
            que = example
            from annotated_demos.auto_annotated.self_ask.annotated import get_rationales

            prediction, program = get_rationales(que, few_prompt, answer=None, follow_up=follow_up)
            results[pid] = {}
            results[pid]["shot_pids"] = shot_pid
            results[pid]["Answer"] = answer
            results[pid]["Program"] = program
            results[pid]["Prediction"] = prediction

            # correct or not
            if answer == prediction:
                correct += 1
                results[pid]["true_false"] = True
            else:
                results[pid]["true_false"] = False

            acc = correct / (i + 1) * 100

            if args.debug:
                logger.write("\n##################################")
                logger.write("[Acc]:\t", results[pid]["true_false"])
                logger.write("[A] labeled answer:\t", answer)
                logger.write("[P] predicted answer:\t", prediction)
                logger.write("[P] generated program:\t", program)
            if count % args.save_every == 0 or count == total or count % 10 == 0:
                if count >= check_count:
                    # have new outputs
                    # file = result_file.split(".")
                    # result_file = file[0] + f"_{count / 10}." + file[1]
                    logger.write(
                        f"{count}/{total}, correct: {correct}, acc: {round(acc, 2)}%, Yes:{i + 1 - shottimes}/{count},saved to {result_file}")
                    save_results(result_file, acc, correct, count, args, results)
                else:
                    # no new outputs, just logger.write the accuracy
                    # file = result_file.split(".")
                    # result_file = file[0] + f"_{count / 10}." + file[1]
                    logger.write(f"{count}/{total}, correct: {correct}, acc: {round(acc, 2)}%, saved to {result_file}")
                    # save_results(result_file, acc, correct, count, cand_pids, args, results)
                    # logger.write(f"{count}/{total}, correct: {correct}, acc: {round(acc, 2)}%")
