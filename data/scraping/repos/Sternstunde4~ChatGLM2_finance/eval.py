import pandas as pd
import json
import re
from pathlib import Path

from transformers import AutoTokenizer, AutoModel
from langchain.embeddings import HuggingFaceEmbeddings
from method.template_manager import template_manager
from method.prompt_generation import (
    get_balance_static, get_balance_sheet_prompt, get_profit_statement_prompt,
    get_cash_flow_statement_prompt, calculate_indicator, GLMPrompt
)
from peft import PeftModel
from transformers import AutoModel

# from finetune import ckpt_path
# from data_preprocess import tokenizer
# model_path = re.sub("\s", "", "D:\LLM_dev\LangChain\model\chatglm2-6b-32k")
# checkpoint = Path(f'{model_path}')
# device = 'cuda:0'
# tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, device=device)
# ckpt_path = 'finance_chatglm2'

# 原模型加上lora参数，保存到chatglm2-6b-finance，作为完整新模型
# model = AutoModel.from_pretrained(checkpoint,
#                                   load_in_8bit=False,
#                                   trust_remote_code=True,
#                                   device_map='auto')
# # model = AutoModel.from_pretrained("D:/LLM_dev/LangChain/model/chatglm2-6b-32k",
# #                                   load_in_8bit=False,
# #                                   trust_remote_code=True,
# #                                   device_map='auto')
# model = PeftModel.from_pretrained(model, ckpt_path)
# model = model.merge_and_unload()  # 合并lora权重
#
# model.save_pretrained("chatglm2-6b-finance", max_shard_size='1GB')
# tokenizer.save_pretrained("chatglm2-6b-finance")

# 使用保存在"chatglm2-6b-finance"中的新模型
model_name = "finetune\chatglm2-6b-finance"
tokenizer = AutoTokenizer.from_pretrained(
    model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().cuda()

# def predict(text, his=[]):
#     response, history = model.chat(tokenizer, f"{text} -> ", history=his, temperature=0.01)
#     return response
#
#
# predict('死鬼，咋弄得这么有滋味呢')
#
# dftest = pd.read_parquet('data/dftest.parquet')
# preds = ['' for x in dftest['text']]
#
# from tqdm import tqdm
# for i in tqdm(range(len(dftest))):
#     text = dftest['text'].loc[i]
#     preds[i] = predict(text)
#
#
# acc = len(dftest.query('tag==pred'))/len(dftest)
# print('acc=',acc)

# device = 'cuda:0'
# model_path = re.sub("\s", "", "D:/LLM_dev/LangChain/model/chatglm2-6b-32k")
# checkpoint = Path(f'{model_path}')
# print(checkpoint)

embeddings = HuggingFaceEmbeddings(model_name="D:/LLM_dev/LangChain/model/text2vec", model_kwargs={'device': "cuda"})
# tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, device=device)
# model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True, device=device).half().cuda()
# model = model.eval()

# model = Model.from_pretrained('ZhipuAI/chatglm2-6b', device_map='auto', revision='v1.0.7')
# pipe = pipeline(task=Tasks.chat, model=model)
COMPUTE_INDEX_SET = [
    '非流动负债比率', '资产负债比率', '营业利润率', '速动比率', '流动比率', '现金比率', '净利润率',
    '毛利率', '财务费用率', '营业成本率', '管理费用率', "企业研发经费占费用",
    '投资收益占营业收入比率', '研发经费与利润比值', '三费比重', '研发经费与营业收入比值', '流动负债比率'
]

def read_questions(path):
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file.readlines()]

def process_question(idx, question_obj):
    glm_prompt = GLMPrompt()

    q = question_obj['question']
    contains_year, year_ = glm_prompt.find_years(q)
    stock_name, stock_info, has_stock = glm_prompt.has_stock(q)
    compute_index = False

    if contains_year and has_stock:
        for t in COMPUTE_INDEX_SET:
            if t in q:
                prompt_res = calculate_indicator(year_[0], stock_name, index_name=t)
                if prompt_res is not None:
                    prompt_ = template_manager.get_template("ratio_input").format(context=prompt_res, question=q)
                    response_, history = model.chat(tokenizer, prompt_, history=[], temperature=0.1)
                    # inputs_t = {'text': prompt_, 'history': []}
                    # response_ = pipe(inputs_t)['response']
                    question_obj["answer"] = str(response_)
                    compute_index = True
                    break
                else:
                    prompt_ = template_manager.get_template("ratio_input").format(context=prompt_res, question=q)
                    response_, history = model.chat(tokenizer, prompt_, history=[], temperature=0.1)
                    # inputs_t = {'text': prompt_, 'history': []}
                    # response_ = pipe(inputs_t)['response']
                    question_obj["answer"] = str(response_)
                    compute_index = True
                    break

    if not compute_index and '增长率' in q:
        statements = [
            get_profit_statement_prompt(q, stock_name, year_),
            get_balance_sheet_prompt(q, stock_name, year_),
            get_cash_flow_statement_prompt(q, stock_name, year_),
            get_balance_static(q, stock_name, year_)
        ]
        prompt_res = [stmt for stmt in statements if len(stmt) > 5]
        if prompt_res:
            prompt_concat = ''
            for prompt_res_obj in prompt_res:
                prompt_concat = prompt_concat + prompt_res_obj
            prompt_ = template_manager.get_template("ratio_input").format(context=prompt_concat, question=q)
            response_, history = model.chat(tokenizer, prompt_, history=[], temperature=0.1)
            # inputs_t = {'text': prompt_, 'history': []}
            # response_ = pipe(inputs_t)['response']
            question_obj["answer"] = str(response_)
            compute_index = True

    if not compute_index:
        prompt_ = glm_prompt.handler_q(q=question_obj['question'])
        response_, history = model.chat(tokenizer, prompt_, history=[], temperature=0.1)
        # inputs_t = {'text': prompt_, 'history': []}
        # response_ = pipe(inputs_t)['response']
        question_obj["answer"] = str(response_)

    with open("result_demo/submit_example_0817.json", "a", encoding="utf-8") as f:
        json.dump(question_obj, f, ensure_ascii=False)
        f.write('\n')
    with open("result_demo/prompts_0817.txt", "a", encoding="utf-8") as f:
        f.write(str(idx) + question_obj['question'] +'\n' + prompt_ + '\n\n')


if __name__ == '__main__':
    questions = read_questions("result_demo/test_questions.jsonl")
    for idx, question_obj in enumerate(questions):
        # if idx > 4938:
        print(f'Processing question {idx}')
        process_question(idx, question_obj)