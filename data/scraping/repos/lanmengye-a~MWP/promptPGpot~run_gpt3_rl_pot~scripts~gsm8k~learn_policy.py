import os
import sys
import math
import json
import argparse
import random
import time
import torch
import openai

import numpy as np
import torch.nn.functional as F

from functools import lru_cache
from run_gpt3_rl_pot.tools import utils
from base_prompt import *
from model import *
from utilities1 import extract_prediction, normalize_answer
from tool import safe_execute,floatify_ans
from utilities1 import get_gpt3_output

sys.path.append("../")
# openai.api_key = "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"

##
def load_data(args):

    problems = [json.loads(line) for line in open("dataset/gsm8k/gsm8k.jsonl","r")]
    demo8 = json.load(open("dataset/gsm8k/demo8_test1.json"))
    cand_pids = [item["Index"] for idx,item in enumerate(demo8) ]
    pids = [idx for idx,item in enumerate(problems) if idx not in cand_pids]
    pids = pids[:500]
    samples = random.sample(pids, args.train_number )  # random sample
    train_pids = samples[:args.train_number]
    return problems, cand_pids, train_pids




def get_batch_reward_loss(scores, cand_pids, pid_batch, label_batch, args):

    batch_loss = 0
    batch_reward = 0
    program,prediction = None,None
    ## loop over the training examples
    for i in range(len(scores)):

        # interact with the environment to get rewards, which in our case is to feed the prompt into GPT-3 and evaluate the prediction
        cand_prob = scores[i, :].clone().detach()
        cand_prob = cand_prob.cpu().numpy()
        cand_prob = np.nan_to_num(cand_prob, nan=0.000001)  # replace np.nan with 0
        cand_prob /= cand_prob.sum()  # make probabilities sum to 1
        # print(f"cand_prob: {cand_prob}")

        # sample shot_pids from the cand_prob distribution
        cids = np.random.choice(range(len(cand_pids)), args.shot_number, p=cand_prob, replace=False)

        # reverse shot_pids so more relevant prompt will be put closer to the question
        cids = cids[::-1]
        # print(f"cids: {cids}")

        shot_pids = [cand_pids[cid] for cid in cids]
        # print(f"shot_pids: {shot_pids}")

        # generate the prompt input
        prompt = build_prompt(problems, shot_pids, pid_batch[i], args)

        # get the output from GPT-3

        unpredicted = get_gpt3_output(prompt, args)
        if unpredicted is isinstance(unpredicted, str):
            program, prediction = None, None
        else:
            try:
                program, prediction = unpredicted

            except Exception as e:
                print(e)
        # print("generated prompt: ", prompt)
        # print("generated program: ", program, prediction)
        # ans = safe_execute(output)
        # prediction = output


        log_prob = 0
        for cid in cids:
            log_prob += torch.log(scores[i, cid])
        # print(f"log_prob: {log_prob}")

        if prediction == eval(label_batch[i]):
            _reward = 1
        else:
            _reward = -1
        # print(f"reward: {reward}")

        batch_reward += _reward
        batch_loss -= _reward * log_prob

    return cids, batch_reward, batch_loss
import matplotlib.pyplot as plt
def plot_dynamic():

    plt.ion()  # 开启交互模式
    plt.figure()  # 创建新的图形窗口
    plt.xlim(0, 100)  # 设置X轴范围
    plt.ylim(0, 10)  # 设置Y轴范围
    plt.xlabel('Batchs')
    plt.ylabel('Loss')
def close_dynamic():
    plt.ioff()  # 关闭交互模式
    plt.show()

def policy_gradient_train( policy_model, problems, train_pids, cand_pids, cand_examples, args):
    # REINFORCE
    # if os.path.exists(args.ckpt_path):
    #     print("!!! Model dir already exists. Consider load it instead of training again.")

    optimizer = torch.optim.Adam(policy_model.parameters(), lr=args.lr)

    train_samples, train_labels, units, options = [], [], [], []
    for pid in train_pids:
        train_samples.append(create_example_from_pid(
            pid, problems, args, test=True))  # Set test=True to avoid answer being added to the training input.
        # answer_norm = normalize_answer(problems[pid]['answer'], problems[pid]['unit'])
        train_labels.append(problems[pid]['Answer'])


    num_batch = math.ceil(len(train_samples) / args.batch_size)

    reward_history = []
    loss_history = []

    total_reward_history = []  # epoch based
    total_loss_history = []  # epoch based

    STOP_FLAG = False
    losses = []
    for epoch in range(args.epochs):
        logger.write(f"Epoch: {epoch}")
        total_train_reward = 0
        total_train_loss = 0

        # We can simply set the batch_size to len(train_data) in few-shot setting.
        for batch_i in range(num_batch):
            logger.write(f"Batch: {batch_i}")
            train_batch = train_samples[batch_i * args.batch_size:(batch_i + 1) * args.batch_size]
            label_batch = train_labels[batch_i * args.batch_size:(batch_i + 1) * args.batch_size]
            pid_batch = train_pids[batch_i * args.batch_size:(batch_i + 1) * args.batch_size]


            # We need to encode cands again every time we update the network
            embedding_cands = policy_model(cand_examples)  # len(cand_examples) x embedding_size
            embedding_ctxt = policy_model(train_batch)  # len(train_batch) x embedding_size

            scores = torch.mm(embedding_ctxt, embedding_cands.t())  # len(train_batch) x len(cand_examples)
            # print(f"unnormed scores: {scores}")

            scores = F.softmax(scores, dim=1)  # len(train_batch) x len(cand_examples)

            cids, reward, loss = get_batch_reward_loss(scores, cand_pids, pid_batch,
                                                       label_batch, args)

            losses.append(loss.item())



            logger.write(f"cids for sample[-1] in batch: {cids}")
            logger.write(f"Cand prob for sample[-1] in batch: {[round(x,5) for x in scores[-1, :].tolist()]}")
            logger.write(f"### reward for the batch: {reward}")
            logger.write(f"### loss for the batch: {loss}\n")

            # linear layer has Weight and bias
            # prev_param = list(policy_model.linear.parameters())[0].clone()
            # print(f"prev_param: {prev_param.data}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # for each iteration/batch
            total_train_reward += reward
            total_train_loss += loss.item()

            reward_history.append(reward)
            loss_history.append(loss.item())

            if np.isnan(loss.item()):
                STOP_FLAG = True
                break

        # 打印输出
        print(f"epoch:{epoch},loss:{total_train_loss}")
        print(f"epoch:{epoch},reward:{total_train_reward}")
        # for each epoch
        total_reward_history.append(total_train_reward)
        total_loss_history.append(total_train_loss)

        best_reward = max(total_reward_history)
        best_loss = min(total_loss_history)

        best_reward_epoch = total_reward_history.index(best_reward)
        best_loss_epoch = total_loss_history.index(best_loss)

        logger.write("============================================")
        logger.write(f"### Epoch: {epoch} / {args.epochs}")
        logger.write(f"### Total reward: {total_train_reward}, " + f"Total loss: {round(total_train_loss,5)}, " +
                     f"Best reward: {best_reward} at epoch {best_reward_epoch}, " +
                     f"Best loss: {round(best_loss, 5)} at epoch {best_loss_epoch}\n")
        # if epoch-best_reward_epoch > 10   or epoch-best_loss_epoch > 2:
        #     STOP_FLAG = True
        #     break
        # save every epoch
        ckpt_file = os.path.join(args.ckpt_path, f"ckpt_{epoch}.pt")
        torch.save(policy_model.linear.state_dict(), ckpt_file)
        logger.write(f"saved the ckpt to {ckpt_file}")

        # save best epoch
        if epoch == best_reward_epoch:
            ckpt_file = os.path.join(args.ckpt_path, "ckpt_best_reward.pt")
            torch.save(policy_model.linear.state_dict(), ckpt_file)
            logger.write(f"saved the best reward ckpt to {ckpt_file}")

        if epoch == best_loss_epoch:
            ckpt_file = os.path.join(args.ckpt_path, "ckpt_best_loss.pt")
            torch.save(policy_model.linear.state_dict(), ckpt_file)
            logger.write(f"saved the best loss ckpt to {ckpt_file}")


        # save reward and loss history
        history = {
            "reward_history": reward_history,
            "loss_history": loss_history,
            "total_reward_history": total_reward_history,
            "total_loss_history": total_loss_history,
        }
        history_file = os.path.join(args.ckpt_path, "history.json")
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2, separators=(',', ': '))

        # print cache info

        logger.write("============================================\n")

        if STOP_FLAG:
            break

    # 绘制曲线
        try:

            plt.plot(total_loss_history, 'b')
            plt.draw()
            plt.pause(0.1)
        except Exception as e:
            print("there is error happening when drawing", e)
    # save in the endclose_dynamic()
    close_dynamic()
    ckpt_file = os.path.join(args.ckpt_path, "ckpt_final.pt")
    torch.save(policy_model.linear.state_dict(), ckpt_file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../data/svamp')
    parser.add_argument('--model', type=str, default='gpt3_rlpot')
    parser.add_argument('--option_inds', type=list, default=["A", "B", "C", "D", "E", "F"])

    # User options
    parser.add_argument('--label', type=str, default='exp1')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument(
        '--prompt_format',
        type=str,
        default='TQ-A',
        choices=['T-A', 'Q-A', 'Q-AS', 'Q-SA', 'TQ-A', 'TQ-AS', 'TQ-SA', 'QT-A', 'QT-AS', 'QT-SA', 'QTS-A', 'TQS-A'],
        help='prompt format template')
    parser.add_argument('--shot_number', type=int, default=2, help='Number of n-shot training examples.')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    # GPT-3 settings
    parser.add_argument('--engine', type=str, default='text-davinci-003', choices=['text-davinci-002', 'ada'])
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max_tokens',
                        type=int,
                        default=512,
                        help='The maximum number of tokens allowed for the generated answer.')
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)

    # Policy gradient settings
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--model_config',
                        type=str,
                        default='bert-base-uncased',
                        choices=['distilbert-base-uncased', 'bert-base-uncased'])
    parser.add_argument('--train_number', type=int, default=20, help='Number of training samples.')
    parser.add_argument('--cand_number', type=int, default=10, help='Number of candidate prompts.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate of policy network.')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs.')
    parser.add_argument('--embedding_size', type=int, default=128, help='Policy network final layer hidden state size.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=2,
                        help='Policy network training batch size. Set to train_number by default.')
    parser.add_argument('--ckpt_root', type=str, default='checkpoints')

    args = parser.parse_args()

    # print and save the args
    args.ckpt_path = os.path.join(args.ckpt_root, args.label)
    utils.create_dir(args.ckpt_path)
    _logger = utils.Logger(args.ckpt_path + '/args.txt')

    print('====Input Arguments====')
    _logger.write(json.dumps(vars(args), indent=2, sort_keys=False))

    return args


if __name__ == '__main__':

    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)  # CPU random seed
    torch.cuda.manual_seed(args.seed)  # GPU random seed
    torch.backends.cudnn.benchmark = True

    ## problems, test question ids, candidate prompt pids, RL training pids
    problems, cand_pids, train_pids = load_data(args)

    ## policy network
    policy_model = policy_network(model_config=args.model_config,
                                  add_linear=True,
                                  embedding_size=args.embedding_size,
                                  freeze_encoder=True)

    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")  # one GPU
    policy_model = policy_model.to(device)

    ## construct candidate examples
    cand_examples = []
    for pid in cand_pids:
        example = create_example_from_pid(pid, problems, args, test=True)
        cand_examples.append(example)

    ## TRAINING
    logger = utils.Logger(os.path.join(args.ckpt_path, 'log.txt'))
    policy_gradient_train(policy_model, problems, train_pids, cand_pids, cand_examples, args)
