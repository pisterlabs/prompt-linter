import json
import random
import ast
from revChatGPT.V1 import Chatbot
import re
import openai
import time

df_access = [
    ('sk-abcd')
]

df_ret = {
    'chinese': {
                'DDD-导致': ['疾病', '疾病'], 'DBD-并发症': ['疾病', '疾病'], 'DHD-合并症': ['疾病', '疾病'], 'DJD-鉴别诊断' : ['疾病', '疾病'], 'DFD-是风险因素': ['疾病', '疾病'], 'DSD-subClassOf': ['疾病', '疾病'], 'TeZD-是诊断依据': ['检查', '疾病'],
                'TeCD-可使用检查': ['检查', '疾病'],'DiODe-有就诊科室': ['疾病', '就诊科室'], 'HeFDi-是风险因素': ['健康危险因素', '疾病'], 'GeRDi-导致': ['微生物', '疾病'], 'DiJDr-禁忌': ['疾病', '药物'], 'DiCDr-采取': ['疾病', '药物'], 'DiZDr-治疗': ['疾病', '药物'],
                'DiEDr-恶化': ['疾病', '药物'], 'DiRDr-导致': ['疾病', '药物'], 'DiRSy-导致': ['疾病', '症状'], 'DiZSy-是诊断依据': ['疾病', '症状'], 'DiLSy-是临床表现': ['疾病', '症状'], 'DiZOp-治疗': ['疾病', '操作'], 'DiEOp-恶化': ['疾病', '操作'], 'DiROp-导致' : ['疾病', '操作'],
                'DiJOp-禁忌': ['疾病', '操作'], 'DiCOp-采取': ['疾病', '操作'], 'DiBBo-有发生部位' : ['疾病', '人体'], 'TeDGe-有作用对象' : ['检查', '微生物'], 'TeJSy-检查结果': ['检查', '症状'], 'TeCSy-可使用检查': ['检查', '症状'], 'TeDBo-有作用对象' : ['检查', '人体'],
                'BoFSy-有发生部位': ['人体', '症状'], 'BoDOp-有作用对象': ['人体', '操作'], 'SyJOp-禁忌': ['症状', '操作'], 'SyCOp-采取': ['症状', '操作'], 'SyZOp-治疗': ['症状', '操作'], 'SyEOp-恶化': ['症状', '操作'], 'SyDOp-导致': ['症状', '操作'], 'SyKDe-有就诊科室': ['症状', '科室'],
                'SyRGe-导致': ['症状', '微生物'], 'SyJDr-禁忌': ['症状', '药物'], 'SyCDr-采取': ['症状', '药物'], 'SyZDr-治疗': ['症状', '药物'], 'SyEDr-恶化': ['症状', '药物'], 'SyRDr-导致': ['症状', '药物'], 'SyBSy-伴随': ['症状', '症状'], 'SyLSy-是临床表现': ['症状', '症状'], 'SyRSy-导致': ['症状', '症状'],
                'SySSy-subClassOf': ['症状', '症状'], 'DrXDr-协同': ['药物', '药物'], 'DrJDr-拮抗': ['药物', '药物'], 'DrSDr-subClassOf' : ['药物', '药物'], 'DrKGe-抗微生物': ['药物', '微生物'], 'SyRHe-导致': ['症状', '健康危险因素'], 'TeSTe-subClassOf': ['检查', '检查'], 'OpSOp-subClassOf': ['操作', '操作'],
                'GeSGe-subClassOf': ['微生物', '微生物'], 'PaSPa-subClassOf': ['人体', '人体'], 'DeSDe-subClassOf': ['科室', '科室'], 'HeSHe-subClassOf': ['健康危险因素', '健康危险因素'], 'TeKDe-有就诊科室': ['检查', '科室'], 'OpKDe-有就诊科室': ['操作', '科室']
    }
}

df_nert = {
    'chinese': ['疾病', '症状', '检查', '药物', '操作', '微生物', '人体', '科室', '健康危险因素'],
    'english': ['disease', 'symptom', 'test', 'drug', 'operation', 'germ', 'body', 'department', 'healthrisk'],
}
    

re_s1_p = {
    'chinese': '''给定的句子为："{}"\n\n给定关系列表：{}\n\n在这个句子中，可能包含了哪些关系？\n请给出关系列表中的关系。\n如果不存在则回答：无\n按照元组形式回复，如 (关系1, 关系2, ……)：''',
    'english': '''The given sentence is "{}"\n\nList of given relations: {}\n\nWhat relations in the given list might be included in this given sentence?\nIf not present, answer: none.\nRespond as a tuple, e.g. (relation 1, relation 2, ......):''',
}

re_s2_p = {
    'chinese': '''根据给定的句子，两个实体的类型分别为（{}，{}）且之间的关系为{}，请找出这两个实体，如果有多组，则按组全部列出。\n如果不存在则回答：无\n按照表格形式回复，表格有两列且表头为（{}，{}）：''',
    'english': '''According to the given sentence, the two entities are of type ('{}', '{}') and the relation between them is '{}', find the two entities and list them all by group if there are multiple groups.\nIf not present, answer: none.\nRespond in the form of a table with two columns and a header of ('{}', '{}'):''',
}

# -------------
ner_s1_p = {
    'chinese': '''给定的句子为："{}"\n\n给定实体类型列表：{}\n\n在这个句子中，可能包含了哪些实体类型？\n如果不存在则回答：无\n按照元组形式回复，如 (实体类型1, 实体类型2, ……)：''',
    'english': '''The given sentence is "{}"\n\nGiven a list of entity types: {}\n\nWhat entity types may be included in this sentence?\nIf not present, answer: none.\nRespond as a list, e.g. [entity type 1, entity type 2, ......]:'''
}

ner_s2_p = {
    'chinese': '''根据给定的句子，请识别出其中类型是"{}"的实体。\n如果不存在则回答：无\n按照表格形式回复，表格有两列且表头为（实体类型，实体名称）：''',
    'english': '''According to the given sentence, please identify the entity whose type is "{}".\nIf not present, answer: none.\nRespond in the form of a table with two columns and a header of (entity type, entity name):'''
}


def chat_re(inda, chatbot):
    print("---RE---")
    mess = [{"role": "system", "content": "You are a helpful assistant."},] # chatgpt对话历史

    # typelist = inda['type']
    typelist = df_ret['chinese']
    sent = inda['sentence']
    lang = inda['lang']

    out = [] # 输出列表 [(e1,r1,e2)]

    try:
        print('---stage1---')
        # 构造prompt
        stage1_tl = list(typelist.keys())
        s1p = re_s1_p[lang].format(sent, str(stage1_tl))
        print(s1p)

        # 请求chatgpt
        mess.append({"role": "user", "content": s1p})
        text1 = chatbot(mess)
        mess.append({"role": "assistant", "content": text1})
        print(text1)

        # 正则提取结果
        res1 = re.findall(r'\(.*?\)', text1)
        print(res1)
        if res1!=[]:
            rels = [temp[1:-1].split(',') for temp in res1]
            rels = list(set([re.sub('[\'"]','', j).strip() for i in rels for j in i]))
            #print(rels)
        else:
            rels = []
        print(rels)
    except Exception as e:
        print(e)
        print('re stage 1 none out or error')
        return ['error-stage1:' + str(e)], mess

    print('---stage2')
    try:
        for r in rels:
            if r in typelist:
                # 构造prompt
                st, ot = typelist[r]
                s2p = re_s2_p[lang].format(st, ot, r, st, ot)
                print(s2p)

                # 请求chatgpt
                mess.append({"role": "user", "content": s2p})
                time.sleep(20)
                text2 = chatbot(mess)
                mess.append({"role": "assistant", "content": text2})
                print(text2)

                # 正则提取结果
                res2 = re.findall(r'\|.*?\|.*?\|', text2)
                print(res2)

                # 进一步处理结果
                count=0
                for so in res2:
                    count+=1
                    if count <=2: # 过滤表头
                        continue

                    so = so[1:-1].split('|')
                    so = [re.sub('[\'"]','', i).strip() for i in so]
                    if len(so)==2:
                        s, o = so
                        #if st in s and ot in o or '---' in s and '---' in o:
                        #    continue 
                        out.append((s, r, o))
                #break
    
    except Exception as e:
        print(e)
        print('re stage 2 none out or error')
        if out == []:
            out.append('error-stage2:' + str(e))
        out = [tup for tup in out if '无' not in tup]
        return out, mess

    if out == []:
        out.append('none-none')
    else:
        out = list(set(out))
        out = [tup for tup in out if '无' not in tup]
    
    print(mess)
    return out, mess

def chat_ner(inda, chatbot):
    print("---NER---")
    mess = [{"role": "system", "content": "You are a helpful assistant."},] # chatgpt对话历史

    # typelist = inda['type']
    typelist = df_nert['chinese']
    sent = inda['sentence']
    lang = inda['lang']

    out = [] # 输出列表 [(e1,et1)]

    try:
        print('---stage1---')
        # 构造prompt
        stage1_tl = typelist
        s1p = ner_s1_p[lang].format(sent, str(stage1_tl))
        print(s1p)

        # 请求chatgpt
        mess.append({"role": "user", "content": s1p})
        text1 = chatbot(mess)
        mess.append({"role": "assistant", "content": text1})
        print(text1)

        # 正则提取结果, ner特殊
        if lang == 'chinese':
            res1 = re.findall(r'\(.*?\)', text1)
        else:
            res1 = re.findall(r'\[.*?\]', text1)
        print(res1)
        if res1!=[]:
            rels = [temp[1:-1].split(',') for temp in res1]
            rels = list(set([re.sub('[\'"]','', j).strip() for i in rels for j in i]))
            #print(rels)
        else:
            rels = []
        print(rels)
    except Exception as e:
        print(e)
        print('ner stage 1 none out or error')
        return ['error-stage1:' + str(e)], mess

    print('---stage2')
    try:
        for r in rels:
            if r in typelist:
                # 构造prompt
                s2p = ner_s2_p[lang].format(r)
                print(s2p)

                # 请求chatgpt
                mess.append({"role": "user", "content": s2p})
                time.sleep(20)
                text2 = chatbot(mess)
                mess.append({"role": "assistant", "content": text2})
                print(text2)

                # 正则提取结果
                res2 = re.findall(r'\|.*?\|.*?\|', text2)
                print(res2)

                # 进一步处理结果
                count=0
                for so in res2:
                    count+=1
                    if count <=2: # 过滤表头
                        continue

                    so = so[1:-1].split('|')
                    so = [re.sub('[\'"]','', i).strip() for i in so]
                    if len(so)==2:
                        s, o = so
                        #if st in s and ot in o or '---' in s and '---' in o:
                        #    continue 
                        out.append((o, r))
    
    except Exception as e:
        print(e)
        print('ner stage 2 none out or error')
        if out == []:
            out.append('error-stage2:' + str(e))
        return out, mess
    

    if out == []:
        out.append('none-none')
    else:
        out = [tup for tup in out if '无' not in tup]
        out = list(set(out))   # 去除重复
    
    print(mess)
    return out, mess

def chat(mess):
    #openai.proxy = 'http://127.0.0.1:10809' # 根据自己服务器的vpn情况设置proxy；如果是在自己电脑线下使用，可以在电脑上开vpn然后不加此句代码。
    #openai.api_base = "https://closeai.deno.dev/v1" 或者利用反向代理openai.com（代理获取：https://github.com/justjavac/openai-proxy）（注释掉上面那句代码）
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=mess
    )

    res = response['choices'][0]['message']['content']
    return res


def chatie(input_data):
    print('input data type:{}'.format(type(input_data)))
    print('input data:{}'.format(input_data))

    # 参数处理，默认参数
    task = input_data['task']
    lang = input_data['lang']
    typelist = input_data['type']
    access = input_data['access']

    ## account
    if access=="":
        print('using default access token')
        tempes = random.choice(df_access)
        input_data['access'] = tempes[1]+tempes[0][1:]

    ## chatgpt
    try:
        openai.api_key = input_data['access']
        chatbot = chat
    except Exception as e:
        print('---chatbot---')
        print(e)
        input_data['result'] = ['error-chatbot']
        return input_data # 没必要进行下去
    
    ## typelist, 空或者出错就用默认的
    try:
        typelist = ast.literal_eval(typelist)
        input_data['type'] = typelist
    except Exception as e:
        print('---typelist---')
        print(e)
        print(typelist)
        print('using default typelist')
        if task == 'RE':
            typelist = df_ret[lang]
            input_data['type'] = typelist
        elif task == 'NER':
            typelist = df_nert[lang]
            input_data['type'] = typelist

    # get output from chatgpt        
    input_data['ner_result'], input_data['mess'] =  chat_ner(input_data, chatbot)
    input_data['re_result'], input_data['mess'] =  chat_re(input_data, chatbot)
    
    print(input_data)

    with open('access_record.json', 'a') as fw:
        fw.write(json.dumps(input_data, ensure_ascii=False)+'\n')

    return input_data

def getReturnData(sen):
    ind = {
        "sentence": sen,
        "type": "",
        "access": "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",
        "task": "",
        "lang": "chinese"
    }
    post_data = chatie(ind)
    return post_data['ner_result'], post_data['re_result']

# if __name__=="__main__":
    
#     p = '''便秘(constipation)是指大便次数减少，一般每周少于3次，伴排便困难、粪便干结。便秘是临床上常见的症状，多长期持续存在，症状扰人，影响生活质量，病因多样，以肠道疾病最为常见，但诊断时应慎重排除其他病因。'''
#     # ind = {
#     #   "sentence": p,
#     #   "type": "",
#     #   "access": "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",
#     #   "task": "RE",
#     #   "lang": "chinese",
#     # }
#     # post_data=chatie(ind)
#     # print(post_data['ner_result'])
#     # print(post_data['re_result'])
#     getReturnData(p)
