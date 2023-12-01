# _*_ coding:utf-8 _*_

"""
@Time: 2022/3/24 8:24 下午
@Author: jingcao
@Email: xinbao.sun@hotmail.com
"""
import copy
import json
import time

import requests
from spider import SpiderConfig



def query_ner(content, type_='XBS'):
    temp = {
        "args": {
            "input_json": {
                "content": "反复咳嗽5天，发热咽喉疼痛2天。",
                "type": "XBS"
            }
        }
    }
    url = 'http://118.31.250.16/struct/api/algorithm/emr_struct'

    req = copy.deepcopy(temp)
    if content.strip():
        req['args']['input_json']['content'] = content
    if type_:
        req['args']['input_json']['type'] = type_

    res = requests.post(url, json=req).json()
    return res


def query_sym_norm(text):
    url = 'http://118.31.52.153:80/api/algorithm/std_norm_api'
    tmpl = {
        "args": {
            "query": "",
            "type": "sym"
        }
    }
    tmpl['args']['query'] = text
    rsp = requests.get(url, json=tmpl).json()
    res = rsp['data']['result']['results']
    if not res:
        return []
    else:
        norm = []
        for sdic in res:
            print()
            norm.append(sdic['norm_res'])
        return norm


def google_translation(queries, dest="zh-CN"):
    """
    注意:  pip install googletrans==3.1.0a0 deprececated
    REF: https://py-googletrans.readthedocs.io/en/latest/
    调用Google翻译 API
    :param query:
    :param dest:
    :return:
    """
    from googletrans import Translator
    trans = Translator()
    dic = {}
    res = trans.translate(queries, src='en', dest=dest)
    for trans_res in res:
        k, v = trans_res.origin, trans_res.text
        dic[k] = v
    return dic

def google_translation2(query, src='en', dest='zh-CN'):
    client = 'client=gtx&dt=t&sl={}&tl={}&q={}'.format(src, dest, query)
    url = "https://translate.googleapis.com/translate_a/single?{}".format(client)
    print(url)
    header = SpiderConfig.get_header()

    request_result = requests.get(url, headers=header)
    translated_text = json.loads(request_result.text)[0][0][0]
    return translated_text

def download_huggingface_dataset(dataset_name, private=False):
    """
    许多用法可参考https://blog.csdn.net/qq_56591814/article/details/120653752
    """
    from datasets import load_dataset
    if private == True:
        dataset_dict = load_dataset(
            dataset_name, # huggingface上对应的name
            use_auth_token='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')
    else:
        dataset_dict = load_dataset(
            dataset_name
        )
    # 从dataset_dict中获取train/test等具体dataset
    dataset = dataset_dict['train'] # 此时Object为Dataset类型
    # dataset.to_csv('保存本地') # 类似to_json(), to_parquet()
    # 对应load_dataset('parquet', data_files={'train': 'xx.parquet'})
    # 或者遍历筛选
    # 或者整体保存至disk dataset.save_to_disc('xx.dataset')
    # 加载 dataset = load_from_disk('xx.dataset')
    # 使用时具体可参考文档

def language_classify(text):
    """
    检测文本语言归属
    """
    # !pip install langid
    import langid
    return langid.classify(text)

def encoding_detect(inp):
    """
    检测文件编码
    """
    import chardet
    with open(inp, 'rb') as f:
        s = f.read()
        res = chardet.detect(s)
        encoding = res.get('encoding')
        return encoding

def test_chatgpt_response():
    import openai
    base_url = "https://api.openai-asia.com/v1"
    key = "sk-7EfWwMczVQIsGk31ybj9dcQCPbJ7Zco52y8TU91eGZHSKOoW" #del last one
    openai.api_base = base_url
    openai.api_key = key
    rsp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content":"请介绍一下你自己"}
        ]
    )
    rsp = json.dumps(rsp, ensure_ascii=False)
    print(rsp)

def chatgpt_api(prompt):
    import openai
    import traceback
    base_url = "https://api.openai-asia.com/v1"
    key = "sk-7EfWwMczVQIsGk31ybj9dcQCPbJ7Zco52y8TU91eGZHSKOoW" #del last one
    openai.api_base = base_url
    openai.api_key = key
    rsp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    try:
        res = rsp['choices'][0]['message']['content']
        if not isinstance(res, str):
            return None
        return res
    except Exception as e:
        err = traceback.format_exc()
        print(err)
    return None



def hallucination_detect(message):
    import openai
    base_url = "https://api.openai-asia.com/v1"
    key = "sk-7EfWwMczVQIsGk31ybj9dcQCPbJ7Zco52y8TU91eGZHSKOoW"  # del last one
    openai.api_base = base_url
    openai.api_key = key
    rsp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message
    )
    return json.dumps(rsp, ensure_ascii=False)

def batch_hallucination_detect():
    import pandas as pd
    df = pd.read_csv('../data_temp/幻觉测试100条.csv')
    lines = []
    df2 = pd.read_csv('../data_temp/test.csv')
    fin_set = set(df2['instruction'])
    for line in df.itertuples():
        _, *args = line
        instruction, output = getattr(line, 'instruction'), getattr(line, 'output')
        if instruction in fin_set:
            continue

        task = "列举上述回答中不符合事实或者hallucination的部分："
        round1 = {
            "role": "user",
            "content": instruction
        }
        round2 = {
            "role": "system",
            "content": output
        }
        round3 = {
            "role": "user",
            "content": task
        }
        msg = [round1, round2, round3]
        try:
            rsp = hallucination_detect(msg)
        except:
            continue
        rsp = json.loads(rsp)
        print(rsp)
        args.append(rsp)
        lines.append(args)
        time.sleep(0.05)
    new_df = pd.DataFrame(lines, columns=list(df.columns) + ['rsp'])
    new_df.to_csv('../data_temp/test2.csv', index=False)
if __name__ == '__main__':

    # dic = google_translation2('hello world')
    # print(dic)
    # test_moderations()
    # test_chatgpt_response()
    # test_moderations()
    r = chatgpt_api('素食主义者是愚蠢的')





