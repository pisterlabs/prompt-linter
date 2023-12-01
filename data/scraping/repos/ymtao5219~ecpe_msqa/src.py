import openai 
import os
import csv
import numpy as np
import pandas as pd
import argparse

def cal_prf(response='response.csv'): 
    # calculate precision, recall and F1
    df = pd.read_csv(response)
    pred_num, acc_num, true_num = 0, 0, 0
    for i in range(len(df)):
        true = eval(df['target'][i])
        pred = eval(df['response'][i])
        correct_pairs = set(true) & set(pred)
        acc_num += len(correct_pairs)
        pred_num += len(pred)
        true_num += len(true)
    p, r = acc_num/(pred_num), acc_num/(true_num)
    f = 2*p*r/(p+r)
    return p, r, f

def extract_pairs_from_txt( annotations_folder='data', annotations_file='all_data_pair.txt'):
    # extract emotion-cause pairs from txt file
    files = {}
    with open(os.path.join('..', annotations_folder, annotations_file), 'r') as csvfile:
        for line in csv.reader(csvfile):
            if len(line) == 1:   
                index = line[0].split()[0]
                number = line[0].split()[-1]
                continue
            if line[0].strip()[0] == '(': 
                target = []
                for i in range(0, len(line), 2):
                    num1 = int(line[i].strip().strip('('))
                    num2 = int(line[i+1].strip().strip(')'))
                    target.append((num1, num2))
                text = []
                continue
            text.append(line)
            document = {'target': target,
                        'text': text,}
            files[index] = document
    return files

def parse_args():
    parser = argparse.ArgumentParser(description='Zeoshot learning with ChatGPT')
    parser.add_argument('--api_keys', type=str, default='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', help='your own API keys')
    args = parser.parse_args()
    return args

if __name__=='__main__':

    # https://platform.openai.com/account/api-keys
    # Load your API key from an environment variable or secret management service
    # openai.api_key = os.getenv("OPENAI_API_KEY")
    #text = "c1: Yesterday morning c2: a policeman visited the old man with the lost money, c3: and told him that the thief was caught. c4: The old man was very happy. c5: But he still feels worried, c6: as he doesn't know how to keep so much money"
    #text = '1,null,null,当 我 看到 建议 被 采纳 2,null,null,部委 领导 写给 我 的 回信 时 3,null,null,我 知道 我 正在 为 这个 国家 的 发展 尽着 一份 力量 4,null,null,27 日 5,null,null,河北省 邢台 钢铁 有限公司 的 普通工人 白金 跃 6,null,null,拿 着 历年来 国家 各部委 反馈 给 他 的 感谢信 7,happiness,激动,激动 地 对 中新网 记者 说 8,null,null,27 年来 9,null,null,国家公安部 国家 工商总局 国家科学技术委员会 科技部 卫生部 国家 发展 改革 委员会 等 部委 均 接受 并 采纳 过 的 我 的 建议'
    #text = '1,null,null,为 尽快 将 女子 救 下 2,null,null,指挥员 立即 制订 了 救援 方案 3,null,null,第一组 在 楼下 铺设 救生 气垫 4,null,null,并 对 周围 无关 人员 进行 疏散 5,null,null,另一组 队员 快速 爬 上 6 楼 6,null,null,在 楼 内 对 女子 进行 劝说 7,null,null,劝说 过程 中 8,null,null,消防官兵 了解 到 9,null,null,该 女子 是 由于 对方 拖欠 工程款 10,null,null,家中 又 急需 用钱 11,null,null,生活 压力 大 12,sadness,无奈,无奈 才 选择 跳楼 轻生'

    args = parse_args()  
    annotations_folder = 'data'
    annotations_file = 'all_data_pair.txt'     
    openai.api_key = args.api_keys
    files = extract_pairs_from_txt(annotations_folder=annotations_folder, annotations_file=annotations_file)
    correct = 0
    num_data = 0
    indexs = []
    targets = []
    responses = []
    for index in files:
        try:
            num_data += 1
            document = ''
            for i in files[index]['text']:    
                document += ', '.join(i) + ' '# #document += ','.join(i).replace(' ', '') + ' '
            document = document.strip()
            target = files[index]['target']
            text_prompt = 'Given this text:' + document + 'Extract emotion-cause pairs and return a list of tuples. In each tuple, the first element is clause numbers of emotion and the second element is clause numbers of emotion. '
            completion = openai.Completion.create(model="text-davinci-003", prompt=text_prompt, temperature=0.7, max_tokens=256)
            response = completion.choices[0].text #'\n\n[(5, 7), (4, 9)]'
            response = eval(response)
            if set(response).intersection(set(target)):
                correct += 1
            targets.append(target)
            responses.append(response)
            indexs.append(index)
            print(num_data, correct)
        except:
            pass

    df = pd.DataFrame({'index': indexs, 'target': targets, 'response': responses})
    df.to_csv('response.csv', index=False)
    print(cal_prf(response='response.csv'))
    


