import tiktoken
import json
from PACKAGE import asyncThread, metric_realization, multi_rouge
import os
import bert_score
from tqdm import tqdm
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
import time
import random

encoding = tiktoken.encoding_for_model('gpt-3.5-turbo-16k')

api_key = ''


def truncate_all(articles, total_max_tokens=30000):
    result = []
    for article in articles:
        result.append(truncate_one(article, total_max_tokens))
    return result


def token_count(text):
    return len(encoding.encode(text))


def truncate_one(article, total_max_tokens=14000):
    result_truncate = []
    divided_articles = article.split('<doc-sep>')
    num_docs = len(divided_articles)
    max_tokens_per_doc = total_max_tokens // num_docs  # Integer division
    for doc in divided_articles:
        tokens = encoding.encode(doc)
        if len(tokens) > max_tokens_per_doc:
            tokens = tokens[:max_tokens_per_doc]
        result_truncate.append(encoding.decode(tokens))
    return '<doc-sep>'.join(result_truncate)


def load_query_article(path):
    filename = 'test.json'
    with open('QMSum/randomIndex/index.json', 'r') as f:
        index_list = json.load(f)
    with open(os.path.join(path, filename), 'r') as f:
        data = json.load(f)
        queries = [item['Query'] for item in data]
    with open(os.path.join(path, 'truncate_one/split_' + filename), 'r') as f:
        articles = json.load(f)
    queries = [queries[index] for index in index_list]
    articles = [articles[index] for index in index_list]
    return queries, articles


def run_truncate():
    path = 'QMSum/dense/MIN'
    wait_process_files = ['test.json']
    for root, dirs, files in os.walk(path):
        # 遍历当前目录下的文件
        for file_name in files:
            # If it is origin file
            if file_name not in wait_process_files:
                continue

            # 检测有没有创建split文件夹
            if not os.path.lexists(os.path.join(root, 'truncate')):
                os.makedirs(os.path.join(root, 'truncate'))

            # Load data
            with open(os.path.join(root, file_name), 'r') as f:
                data = json.load(f)
                articles = [item['Article'] for item in data]
            split_meetings = truncate_all(articles, 14000)

            # Write meetings
            with open(os.path.join(root, 'truncate_one/split_' + file_name), 'w') as f:
                print(root)
                temp = json.dumps(split_meetings, indent=4)
                f.write(temp)


def intermediate_summary(query, doc):
    map_prompts = f"Write an answer based on the following question and the given meeting.Try to answer thoroughly and do not leave out useful information.\n MEETING:{doc}\n QUESTION:{query}\n SUMMARY: \n"

    # map_prompts = [
    #     f"Abstract the paragraph from the meeting which can be used to answer the question. Do not leave out useful information.\n MEETING:{doc}\n QUESTION:{query}\n ABSTRACTED PARAGRAPH: \n"
    #     for doc in docs]
    system = "You are a helpful assistant that gives long answer to question based on a long meeting."
    messages = [[{"role": "system", "content": system},
                 {"role": "user", "content": map_prompts}]]
    intermediate_outputs = asyncThread.run(messages=messages,
                                           engine_name="gpt-3.5-turbo-16k-0613",

                                           temperature=0.7,
                                           max_tokens=600,
                                           top_p=0.9,
                                           api_key=api_key,
                                           requests_per_minute=20)

    return intermediate_outputs


def traverse_sub_path(path):
    queries, articles = load_query_article(path)

    save_intermediate_outputs = []

    for index, article in enumerate(articles):
        # 运行并处理
        query = queries[index]
        intermediate_outputs = intermediate_summary(query, article)
        save_intermediate_outputs.append(intermediate_outputs)
        with open(os.path.join(path, 'truncate_one/gpt3_summary.json'), 'w') as f:
            temp = json.dumps(save_intermediate_outputs, indent=4)
            f.write(temp)


class LoadEvaluateData:
    @staticmethod
    def load_pred(path, model_name):
        predictions = []
        # pred_file = model_name + '_intermediate_summary.json'
        pred_file = model_name + '_summary.json'
        file_path = os.path.join(path, 'truncate_one/' + pred_file)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                predictions = json.load(f)

        # predictions = [item[0] for item in predictions]

        return predictions

    @staticmethod
    def load_ref(path):
        ref_file = 'test.json'
        references = []
        file_path = os.path.join(path, ref_file)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                references = json.load(f)
            references = [
                data_item['Summary']
                for data_item in references]
        return references


class Evaluate:
    @staticmethod
    def squality_rouge(path, predictions, references, model_name):
        # Calculate
        print('Evaluate rouge score (use squality)')
        rouge_object = multi_rouge.Rouge()
        squality_rouge_score = rouge_object._compute(predictions=predictions,
                                                     references=[[item] for item in references], use_stemmer=True)
        # Save
        file_name = model_name + '_squality_rouge.json'
        # file_name = model_name + '_truncate_squality_rouge.json'
        file_path = os.path.join(path, 'truncate_one/' + file_name)
        with open(file_path, 'w') as f:
            f.write(str(squality_rouge_score))

    @staticmethod
    def gpt_eval(path, predictions, references, model_name):
        # Get prompt
        # metric_list = ['coh', 'con', 'flu', 'rel']
        metric_list = ['con', 'rel']
        for metric_type in metric_list:
            prompt = open('GPTeval/prompts/' + metric_type + '_detailed.txt').read()

            # Get messages
            messages = []
            for index, prediction in enumerate(predictions):
                reference = references[index]
                cur_prompt = prompt.replace('{{Document}}', reference).replace('{{Summary}}', prediction)
                messages.append([{"role": "system", "content": cur_prompt}])

            response_13list = []
            for _ in range(23):
                print(_)
                # Send request
                response_list = asyncThread.run(messages=messages,
                                                engine_name="gpt-3.5-turbo-16k-0613",

                                                temperature=1,
                                                max_tokens=5,
                                                top_p=1,
                                                api_key=api_key,
                                                requests_per_minute=180)

                # Del non-numeric
                num_list = ['1', '2', '3', '4', '5']
                response_list = [item for item in response_list if item and item[0] in num_list]
                response_list = [int(item[0]) for item in response_list]

                response_13list.extend(response_list)

            # Calaulate Average
            average = [sum(response_13list) / len(response_13list)]

            # Save
            # save_path = os.path.join(path, 'evaluation/' + model_name + '_truncate_' + metric_type + '_gpteval.json')
            save_path = os.path.join(path, 'truncate_one/' + model_name + '_' + metric_type + '_gpteval.json')
            gpteval = {'Summary': response_13list, 'average': average}
            with open(save_path, 'w') as f:
                temp = json.dumps(gpteval)
                f.write(temp)

    @staticmethod
    def evaluate(path, model_name, bert=False, rouge=False, another_rouge=False, bleurt=False, gpteval=False):
        # Load predictions
        predictions = LoadEvaluateData.load_pred(path, model_name)
        if not predictions:
            return
        # Load references
        references = LoadEvaluateData.load_ref(path)
        # Load random index
        with open('QMSum/randomIndex/index.json', 'r') as f:
            random_index_list = json.load(f)
        # Change references same to prediciton
        if model_name.startswith('gpt3'):
            references = [references[index] for index in random_index_list]
        # predictions = [predictions[index] for index in random_index_list]

        # Delete empty
        references = [references[index] for index, item in enumerate(predictions) if item != '']
        predictions = [predictions[index] for index, item in enumerate(predictions) if item != '']

        if rouge:
            Evaluate.squality_rouge(path, predictions, references, model_name)
        if another_rouge:
            Evaluate.another_rouge(path, predictions, references, model_name)
        if bert:
            Evaluate.bert(path, predictions, references, model_name)
        if bleurt:
            Evaluate.bleurt(path, predictions, references, model_name)
        if gpteval:
            Evaluate.gpt_eval(path, predictions, references, model_name)


# traverse_sub_path('QMSum/sparse/MIN')
Evaluate.evaluate('QMSum/sparse/MIN', 'gpt3', gpteval=True, rouge=True)
