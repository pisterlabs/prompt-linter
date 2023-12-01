# Setup environment value
import os
from dotenv import load_dotenv
load_dotenv()


## anthropic
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
anthropic_api_key = os.getenv("anthropic_api_key")
anthropic = Anthropic(api_key=anthropic_api_key)

## openai
import openai
openai_api_key = os.getenv("openai_api_key")
openai_gpt_model = "gpt-3.5-turbo-16k"

openai.api_key = openai_api_key


## RoBERTa
# https://huggingface.co/uer/roberta-base-chinese-extractive-qa
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
RoBERTa_model = AutoModelForQuestionAnswering.from_pretrained('uer/roberta-base-chinese-extractive-qa')
RoBERTa_tokenizer = AutoTokenizer.from_pretrained('uer/roberta-base-chinese-extractive-qa')
RoBERTa_QA = pipeline('question-answering',model=RoBERTa_model, tokenizer=RoBERTa_tokenizer)

## bert
# https://huggingface.co/NchuNLP/Chinese-Question-Answering
from transformers import BertTokenizerFast, BertForQuestionAnswering, pipeline
bert_model_name = "NchuNLP/Chinese-Question-Answering"
bert_tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)
bert_model = BertForQuestionAnswering.from_pretrained(bert_model_name)
bert_nlp = pipeline('question-answering', model=bert_model, tokenizer=bert_tokenizer)

def qa_by_anthropic(source_content, question):
    try:
        completion = anthropic.completions.create(
            model="claude-2",
            max_tokens_to_sample=300,
            prompt= f"""Human: 
                我會給你一份檔案。 然後我會向你提問， 利用檔案內的內容來回答。 這是檔案內容：      
                {source_content}
                \n
                採用我提供的資料用繁體中文嘗試定義或回答：{question}，如果發現內容無法回答則回覆「無法提供最佳答案」。
                這是單次問答無須說明開頭與結尾 \nAssistant:
            """.encode('utf-8'),
        )
        return {
            "state": True,
            "value": completion.completion,
        }
    except Exception as exp:
        return {
            "state": False,
            "value": str(exp),
        }
    

def qa_by_openai(source_content,question):
    try:
        # https://platform.openai.com/docs/models/gpt-3-5
        # https://beta.openai.com/docs/api-reference/completions/create
        response = openai.ChatCompletion.create(
            model = openai_gpt_model,
            messages= [
                {"role": "system", "content": f"我會給你一份檔案。 然後我會向你提問， 利用檔案內的內容來回答。如果發現內容無法回答則回覆「無法提供最佳答案」。這是檔案內容：{source_content}" },
                {"role": "user", "content": f"採用我提供的資料用繁體中文嘗試回答：{question}，這是單次問答無須說明開頭與結尾。" }
            ],
            temperature=0, # 嘗試固定回答
            max_tokens=500, # 最多回答 500 tokens
        )
        return {
            "state": True,
            "value": response['choices'][0]['message']['content'],
        }
    except  Exception as exp:
        return {
            "state": False,
            "value": str(exp),
        }


def qa_by_RoBERTa(source_content,question):
    try:
        RoBERTa_QA_input = {'question': "利用文件解答" + question, 'context': source_content}
        result = RoBERTa_QA(RoBERTa_QA_input)
        return {
            "state": True,
            "value": result["answer"],
        }
    except Exception as exp:
        return {
            "state": False,
            "value": str(exp),
        }


def qa_by_bert(source_content,question):
    try:
        bert_QA_input = {
            'question': question,
            'context': source_content
        }
        result = bert_nlp(bert_QA_input)
        return {
            "state": True,
            "value": f"{result['answer']} [{result['score']}]",
        }
    except Exception as exp:
        return {
            "state": False,
            "value": str(exp),
        }




