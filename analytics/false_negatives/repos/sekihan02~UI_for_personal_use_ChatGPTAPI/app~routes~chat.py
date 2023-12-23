import uuid
import os
import ast
import io
import json
# import base64
import litellm
import tokentrim as tt
from typing import List, Dict, Union
import sys
import ast

import requests
from tqdm import tqdm
from glob import glob
import numpy as np
import pandas as pd

from PyPDF2 import PdfReader
import pdfplumber

from urllib.parse import unquote
from flask import Blueprint, render_template
from flask import Flask, request, jsonify
from flask import current_app as app
from flask import stream_with_context, Response
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import tiktoken

import wikipedia
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain

from sklearn.metrics.pairwise import cosine_similarity as cs

chat_bp = Blueprint('chat', __name__)

# Global settings for model and temperature
settings = {
    'model': 'gpt-3.5-turbo-0613',
    'temperature': 0.8,
    'api_key': '',
    'bing_search_v7_subscription_key': '',
    'bing_search_v7_endpoint': 'https://api.bing.microsoft.com',
    'search_service_name': 'text-embedding-ada-002',
    'index_name': 'pdf-1page',
    'strage_search_key': '',
}

# Making Recommendationsのチェック取得
should_recommend = False  # Initial value
# Open Interpreterのチェック取得
should_open_interpreter = False  # Initial value
# Making Wiki Recommendationsのチェック取得
should_recommend_wiki = False  # Initial value
# Making Bing Search Recommendationsのチェック取得
should_recommend_bing = False  # Initial value
# Making Bing Recommendationsのチェック取得
should_recommend_rec_bing = False  # Initial value
# Enable Strage Searchのチェック取得
should_strage_search = False  # Initial value

@chat_bp.route("/")
def index():
    return render_template('index.html')

# A dictionary to manage multiple chat sessions
chat_sessions = {}
# Generate a new session ID (for simplicity, we'll use an incremental integer)
current_session_id = 0

# @chat_bp.route('/start_new_session', methods=['GET'])
# def start_new_session():
#     global current_session_id
#     chatLog = chat_sessions.get(current_session_id, [])
    
#     # Store the current chat log to the session list
#     if chatLog:
#         chat_sessions[current_session_id] = chatLog
#     current_session_id = str(uuid.uuid4())
#     chat_sessions[current_session_id] = []
    
#     return jsonify(session_id=current_session_id)

@chat_bp.route('/start_new_session', methods=['GET']) 
def start_new_session():
    global current_session_id 
    current_session_id += 1 
    chat_sessions[current_session_id] = [] 
    # Initialize a new chat session return 
    jsonify(session_id=current_session_id) 
    
@chat_bp.route('/delete_session/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    if session_id in chat_sessions:
        del chat_sessions[session_id]
    return jsonify(success=True)

@chat_bp.route('/update_settings', methods=['POST'])
def update_settings():
    data = request.json
    settings['model'] = data['model']
    temperature = float(data['temperature'])
    if 0.0 <= temperature <= 1.0:
        settings['temperature'] = temperature
    else:
        return jsonify({'status': 'error', 'message': 'Temperature must be between 0.0 and 1.0'})

    settings['api_key'] = data['api_key']
    # 追加
    settings['bing_search_v7_subscription_key'] = data['bing_search_v7_subscription_key']
    settings['bing_search_v7_endpoint'] = data['bing_search_v7_endpoint']
    settings['search_service_name'] = data['search_service_name']
    settings['index_name'] = data['index_name']
    settings['strage_search_key'] = data['strage_search_key']
    
    return jsonify({'status': 'success'})

@chat_bp.route('/update_recommendation', methods=['POST'])
def update_recommendation():
    global should_recommend
    data = request.json
    should_recommend = data['should_recommend']
    # app.logger.info(f"Updated should_recommend: {should_recommend}")  # Debugging statement
    return jsonify({'status': 'success'})

@chat_bp.route('/update_wiki_recommendation', methods=['POST'])
def update_wiki_recommendation():
    global should_recommend_wiki
    data = request.json
    should_recommend_wiki = data['should_recommend_wiki']
    # app.logger.info(f"Updated should_recommend_wiki: {should_recommend_wiki}")  # Debugging statement
    return jsonify({'status': 'success'})

# 追加
@chat_bp.route('/update_bing_recommendation', methods=['POST'])
def update_bing_recommendation():
    global should_recommend_bing
    data = request.json
    should_recommend_bing = data['should_recommend_bing']
    # app.logger.info(f"Updated should_recommend_bing: {should_recommend_bing}")  # Debugging statement
    return jsonify({'status': 'success'})

@chat_bp.route('/update_rec_bing_recommendation', methods=['POST'])
def update_rec_bing_recommendation():
    global should_recommend_rec_bing
    data = request.json
    should_recommend_rec_bing = data['should_recommend_rec_bing']
    return jsonify({'status': 'success'})

@chat_bp.route('/update_strage_search', methods=['POST'])
def update_strage_search():
    global should_strage_search
    data = request.json
    should_strage_search = data['should_strage_search']
    return jsonify({'status': 'success'})

@chat_bp.route('/open_interpreter', methods=['POST'])
def open_interpreter():
    global should_open_interpreter
    data = request.json
    should_open_interpreter = data['should_open_interpreter']
    return jsonify({'status': 'success'})


class StreamingLLMMemory:
    """
    StreamingLLMMemory クラスは、最新のメッセージと特定数のattention sinksを
    メモリに保持するためのクラスです。
    
    attention sinksは、言語モデルが常に注意を向けるべき初期のトークンで、
    モデルが過去の情報を"覚えて"いるのを手助けします。
    """
    def __init__(self, max_length=10, attention_sinks=4):
        """
        メモリの最大長と保持するattention sinksの数を設定
        
        :param max_length: int, メモリが保持するメッセージの最大数
        :param attention_sinks: int, 常にメモリに保持される初期トークンの数
        """
        self.memory = []
        self.max_length = max_length
        self.attention_sinks = attention_sinks
    
    def get(self):
        """
        現在のメモリの内容を返します。
        
        :return: list, メモリに保持されているメッセージ
        """
        return self.memory
    
    def add(self, message):
        """
        新しいメッセージをメモリに追加し、メモリがmax_lengthを超えないように
        調整します。もしmax_lengthを超える場合、attention_sinksと最新のメッセージを
        保持します。
        
        :param message: str, メモリに追加するメッセージ
        """
        self.memory.append(message)
        if len(self.memory) > self.max_length:
            self.memory = self.memory[:self.attention_sinks] + self.memory[-(self.max_length-self.attention_sinks):]
    
    # def add_pair(self, user_message, ai_message):
    def add_pair(self, user_message, ai_message):
        """
        ユーザーとAIからのメッセージのペアをメモリに追加します。
        
        :param user_message: str, ユーザーからのメッセージ
        :param ai_message: str, AIからのメッセージ
        """
        # self.add("User: " + user_message)
        # self.add("AI: " + ai_message)
        self.add({"role": "user", "content": user_message})
        self.add({"role": "assistant", "content": ai_message})
    
    # ここにはStreamingLLMとのインタラクションのための追加のメソッドを
    # 実装することもできます。例えば、generate_response, update_llm_modelなどです。

# 16件のメッセージを記憶するように設定
memory = StreamingLLMMemory(max_length=16)

# encoding = tiktoken.encoding_for_model(settings['model'])
output_counter = 0  # Add a global counter for output
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# 進行中の生成を停止するためのフラグ
stop_generation_flag = False

@chat_bp.route('/stop_generation', methods=['POST'])
def stop_generation():
    global stop_generation_flag
    stop_generation_flag = True
    return jsonify({'status': 'success', 'message': 'Generation stopping'})


@chat_bp.route('/get_response', methods=['POST'])
def get_response():
    global stop_generation_flag
    stop_generation_flag = False  # フラグをリセット
    # get the user's message from the POST request
    message = request.get_json()['message']

    # 以前の会話をメモリから取得
    past_conversation = memory.get()

    # Get the response from the GPT model
    openai.api_key = settings['api_key']
    os.environ["OPENAI_API_KEY"] = openai.api_key
    llm = ChatOpenAI(
        model_name=settings['model'],
        temperature=settings['temperature']
    )
    template = """あなたは質問に対して、回答を返してください
    質問:{question}
    回答:"""

    # プロンプトテンプレートの生成
    q_template = PromptTemplate(
        input_variables=["question"], 
        template=template
    )

    # LLMChainの準備
    q_chain = LLMChain(
        llm=llm, 
        prompt=q_template, 
        output_key="res"
    )

    res_chain = SequentialChain(
        chains=[q_chain],
        input_variables=["question"], 
        output_variables=["res"],
        verbose=True
    )
    
    # Open Interpreterのチェックがあったら
    if should_open_interpreter:
        print(should_open_interpreter)
        # Paste your OpenAI API key below.
        # Let's try to use the updated Interpreter class
        interpreter = Interpreter()
        res, counter = interpreter.chat(message)

        # counter += num_tokens_from_string(res, settings['model'])
        # counter += num_tokens_from_string("\n".join(interpret), settings['model'])
    else:
        conversations = past_conversation
        # 指示文をメモリの内容に追加
        conversations.append({"role": "system", "content": "Please answer the question."})
        # 新しい質問をメモリの内容に追加
        conversations.append({"role": "user", "content": message})        

        # メモリに新しい会話のペアを追加
        def generate_response():
            try:
                print("get_response endpoint called")  # この行を追加

                response = openai.ChatCompletion.create(
                    model=settings['model'],
                    messages=conversations,
                    temperature=settings['temperature'],
                    stream=True
                )
                
                app.logger.debug("Response from OpenAI: %s", response)  # ログにデバッグ情報を出力
                
                for chunk in response:
                    if stop_generation_flag:
                        print("Generation stopped by user")
                        break  # ユーザーによって停止された場合はループを抜ける
                    chunk_str = json.dumps(chunk).encode('utf-8') + b'\n'
                    app.logger.debug("Yielding chunk: %s", chunk_str) 
                    yield chunk_str

            except Exception as e:
                print("Error calling OpenAI API:", str(e))
                app.logger.error("Error calling OpenAI API: %s", str(e))  # エラー情報をログに出力
                
    return Response(stream_with_context(generate_response()), content_type='application/json')

@chat_bp.route('/process_sync', methods=['POST'])
def process_sync():
    global should_recommend
    global should_recommend_wiki
    global should_recommend_bing  # 追加
    global should_recommend_rec_bing  # 追加
    global should_strage_search  # 追加
    global output_counter  # 追加
    global should_open_interpreter  # 追加
    counter = 0

    res = ""
    recommendations = ""
    rec_bing_search = ""
    strage_search = ""
    bing_search = ""
    wiki_search = ""
    
    data = request.json
    message = data.get('message', "")
    res = data.get('response', "")
    counter += num_tokens_from_string(message + res, settings['model'])
    
    app.logger.debug("Debug message: %s", message) 
    app.logger.debug("Debug responce: %s", res) 
    # Get the response from the GPT model
    openai.api_key = settings['api_key']
    os.environ["OPENAI_API_KEY"] = openai.api_key
    llm = ChatOpenAI(
        model_name=settings['model'],
        temperature=settings['temperature']
    )
    
    # RecommendのMaking Recommendationsのチェックがあったら
    if should_recommend:
        rec_template = """あなたは回答を入力として受け取り、その回答を元に次に質問したり、問い合わせたりした方がいい質問を5つ箇条書きで生成してください
        回答:{response}
        質問"""

        # プロンプトテンプレートの生成
        rec_temp = PromptTemplate(
            input_variables=["response"], 
            template=rec_template
        )

        # LLMChainの準備
        rec_chain = LLMChain(
            llm=llm, 
            prompt=rec_temp, 
            output_key="recommend"
        )

        res_recchain = SequentialChain(
            chains=[rec_chain],
            input_variables=["response"], 
            output_variables=["recommend"],
            verbose=True
        )
        q_recommend = res_recchain({"response": res})
        recommendations = ["次に推奨される質問は次のようなものが考えられます。"] + q_recommend["recommend"].split('\n')
        
        app.logger.debug("Debug q_recommend: %s", q_recommend)
        app.logger.debug("Debug recommendations: %s", recommendations)

        counter += num_tokens_from_string(rec_temp.format(response=["次に推奨される質問は次のようなものが考えられます。"] + q_recommend["recommend"].split('\n')), settings['model'])
        counter += num_tokens_from_string(q_recommend["recommend"], settings['model'])
        # return jsonify({'response': response['res'], 'recommendations': recommendations})

    # Bing Suggestのチェックがあったら
    if should_recommend_rec_bing:
        word_list_template = """
        以下が回答を3つのキーワードに分割した例です。
        ---
        回答: - 寿司
        - ラーメン
        - カレーライス
        - ピザ
        - 焼肉
        キーワード: 寿司 ラーメン カレーライス
        ---
        ---
        回答: 織田信長は、戦国時代の日本で活躍した武将・戦国大名です。信長は、尾張国の織田家の当主として生まれ、若い頃から戦国時代の混乱を乗り越えて勢力を拡大しました。政治的な手腕も備えており、国内の統一を目指し、戦国大名や寺社などとの同盟を結びました。彼の統一政策は、後の豊臣秀吉や徳川家康による天下統一に繋がっていきました。
        信長の死は、本能寺の変として知られています。彼は家臣の明智光秀によって襲撃され、自害に追い込まれました。しかし、彼の業績や影響力は、その後の日本の歴史に大きく残りました。
        キーワード: 織田信長 戦国時代 本能寺
        ---
        回答:{response}
        キーワード"""

        # プロンプトテンプレートの生成
        word_list_temp = PromptTemplate(
            input_variables=["response"], 
            template=word_list_template
        )

        # LLMChainの準備
        word_list_chain = LLMChain(
            llm=llm, 
            prompt=word_list_temp, 
            output_key="keywords"
        )

        keyword_recchain = SequentialChain(
            chains=[word_list_chain],
            input_variables=["response"], 
            output_variables=["keywords"],
            verbose=True
        )
        keywords = keyword_recchain({"response": res})
        # 文字列をPythonのリストに変換
        keyword_list = keywords["keywords"].split(' ')
        
        bing_template = """あなたは回答と検索結果の内容を入力として受け取り、回答と検索結果を参考に次にするべき質問を5以上生成してください。
        生成結果の先頭は必ず順番に1. 2. と数字を必ず記載して生成してください。
        回答:{response}
        検索結果の内容:{bing_search}
        質問"""
        
        # プロンプトテンプレートの生成
        bing_temp = PromptTemplate(
            input_variables=["response", "bing_search"], 
            template=bing_template
        )

        # LLMChainの準備
        bing_chain = LLMChain(
            llm=llm, 
            prompt=bing_temp, 
            output_key="summary_list"
        )

        summary_recchain = SequentialChain(
            chains=[bing_chain],
            input_variables=["response", "bing_search"], 
            output_variables=["summary_list"],
            verbose=True
        )

        # 各キーワードについてBing検索を実行し、結果の要約する
        results = get_bing_search_results_for_keywords(keyword_list, num_results=3, lang='ja-JP')
        
        # キーワードごとにスニペットをグループ化
        grouped_results = {}
        for result in results:
            keyword = result['keyword']
            snippet = result['snippet']
            
            # キーワードがgrouped_resultsにない場合は追加
            if keyword not in grouped_results:
                grouped_results[keyword] = []
            
            # キーワードに対応するリストにスニペットを追加
            grouped_results[keyword].append(snippet)

        # 各キーワードのスニペットを、セパレーターを使って連結
        concatenated_snippets_list = []
        separator = " ---- "  # 区切り文字として意味のある文字列を選択
        for keyword, snippets in grouped_results.items():
            concatenated_snippets = separator.join(snippets)
            concatenated_snippets_list.append(concatenated_snippets)
            
        summary_bing = summary_recchain({"response": res, "bing_search": concatenated_snippets_list})
        
        rec_bing_search = ["次に推奨される質問は次のようなものが考えられます。"] + summary_bing["summary_list"].split('\n')
        # rec_bing_search = ["次に推奨される質問は次のようなものが考えられます。"] + "てすと\nです\ntest\ntest\ntest".split('\n')

        # return jsonify({'response': response['res'], 'rec_bing_search': rec_bing_search})

        counter += num_tokens_from_string(bing_temp.format(response=res, bing_search=concatenated_snippets_list), settings['model'])
        counter += num_tokens_from_string(summary_bing["summary_list"], settings['model'])
    # Bing Searchのチェックがあったら
    if should_recommend_bing:    
        word_list_template = """
        以下が回答を3つのキーワードに分割した例です。
        ---
        回答: - 寿司
        - ラーメン
        - カレーライス
        - ピザ
        - 焼肉
        キーワード: 寿司 ラーメン カレーライス
        ---
        ---
        回答: 織田信長は、戦国時代の日本で活躍した武将・戦国大名です。信長は、尾張国の織田家の当主として生まれ、若い頃から戦国時代の混乱を乗り越えて勢力を拡大しました。政治的な手腕も備えており、国内の統一を目指し、戦国大名や寺社などとの同盟を結びました。彼の統一政策は、後の豊臣秀吉や徳川家康による天下統一に繋がっていきました。
        信長の死は、本能寺の変として知られています。彼は家臣の明智光秀によって襲撃され、自害に追い込まれました。しかし、彼の業績や影響力は、その後の日本の歴史に大きく残りました。
        キーワード: 織田信長 戦国時代 本能寺
        ---
        回答:{response}
        キーワード"""

        # プロンプトテンプレートの生成
        word_list_temp = PromptTemplate(
            input_variables=["response"], 
            template=word_list_template
        )

        # LLMChainの準備
        word_list_chain = LLMChain(
            llm=llm, 
            prompt=word_list_temp, 
            output_key="keywords"
        )

        keyword_recchain = SequentialChain(
            chains=[word_list_chain],
            input_variables=["response"], 
            output_variables=["keywords"],
            verbose=True
        )
        keywords = keyword_recchain({"response": res})
        # 文字列をPythonのリストに変換
        keyword_list = keywords["keywords"].split(' ')
        
        bing_template = """あなたは検索結果の内容を入力として受け取り、要約を最大で5つ箇条書きで生成してください。
        生成結果の先頭は必ず順番に1. 2. と数字を必ず記載して生成してください。
        検索結果の内容:{bing_search}
        要約"""
        
        # プロンプトテンプレートの生成
        bing_temp = PromptTemplate(
            input_variables=["bing_search"], 
            template=bing_template
        )

        # LLMChainの準備
        bing_chain = LLMChain(
            llm=llm, 
            prompt=bing_temp, 
            output_key="summary_list"
        )

        summary_recchain = SequentialChain(
            chains=[bing_chain],
            input_variables=["bing_search"], 
            output_variables=["summary_list"],
            verbose=True
        )

        # 各キーワードについてBing検索を実行し、結果の要約する
        results = get_bing_search_results_for_keywords(keyword_list, num_results=3, lang='ja-JP')
        
        # キーワードごとにスニペットをグループ化
        grouped_results = {}
        for result in results:
            keyword = result['keyword']
            snippet = result['snippet']
            
            # キーワードがgrouped_resultsにない場合は追加
            if keyword not in grouped_results:
                grouped_results[keyword] = []
            
            # キーワードに対応するリストにスニペットを追加
            grouped_results[keyword].append(snippet)

        # 各キーワードのスニペットを、セパレーターを使って連結
        concatenated_snippets_list = []
        separator = " ---- "  # 区切り文字として意味のある文字列を選択
        for keyword, snippets in grouped_results.items():
            concatenated_snippets = separator.join(snippets)
            concatenated_snippets_list.append(concatenated_snippets)
            
        summary_bing = summary_recchain({"bing_search": concatenated_snippets_list})
        
        bing_search = ["関連ワードを調査しました。"] + summary_bing["summary_list"].split('\n')

        # return jsonify({'response': response['res'], 'bing_search': bing_search})

        counter += num_tokens_from_string(bing_temp.format(bing_search=concatenated_snippets_list), settings['model'])
        counter += num_tokens_from_string(summary_bing["summary_list"], settings['model'])
        
    # WikiSearchのMaking Wiki Searchのチェックがあったら
    if should_recommend_wiki:
        list_template = """あなたは回答を入力として受け取り、回答を表す3つ単語に変換してください。
        以下が単語リストの生成例です。
        ---
        回答: - 寿司
        - ラーメン
        - カレーライス
        - ピザ
        - 焼肉
        単語リスト: 寿司 ラーメン カレーライス
        ---
        ---
        回答: 織田信長は、戦国時代の日本で活躍した武将・戦国大名です。信長は、尾張国の織田家の当主として生まれ、若い頃から戦国時代の混乱を乗り越えて勢力を拡大しました。政治的な手腕も備えており、国内の統一を目指し、戦国大名や寺社などとの同盟を結びました。彼の統一政策は、後の豊臣秀吉や徳川家康による天下統一に繋がっていきました。
        信長の死は、本能寺の変として知られています。彼は家臣の明智光秀によって襲撃され、自害に追い込まれました。しかし、彼の業績や影響力は、その後の日本の歴史に大きく残りました。
        単語リスト: 織田信長 戦国時代 本能寺
        ---
        回答:{response}
        単語リスト"""

        # プロンプトテンプレートの生成
        list_temp = PromptTemplate(
            input_variables=["response"], 
            template=list_template
        )

        # LLMChainの準備
        list_chain = LLMChain(
            llm=llm, 
            prompt=list_temp, 
            output_key="lang_list"
        )

        list_recchain = SequentialChain(
            chains=[list_chain],
            input_variables=["response"], 
            output_variables=["lang_list"],
            verbose=True
        )
        list_lang = list_recchain({"response": res})
        # 文字列をPythonのリストに変換
        lang_list = list_lang["lang_list"].split(' ')
        
        # wikiの単語リストの内容取得
        articles = get_wikipedia_articles_for_keywords(lang_list, num_articles=1, lang='ja')
        # 使用する最大トークン数
        # MAX_TOKENS = 4096
        MAX_TOKENS = 1000
        articles_content = [article['content'][:MAX_TOKENS] for article in articles]
        
        wiki_template = """あなたは検索結果の内容を入力として受け取り、要約を最大で5つ箇条書きで生成してください。
        生成結果の先頭は必ず順番に1. 2. と数字を必ず記載して生成してください。
        検索結果の内容:{wiki_search}
        要約"""
        
        # プロンプトテンプレートの生成
        wiki_temp = PromptTemplate(
            input_variables=["wiki_search"], 
            template=wiki_template
        )

        # LLMChainの準備
        wiki_chain = LLMChain(
            llm=llm, 
            prompt=wiki_temp, 
            output_key="summary_list"
        )

        summary_recchain = SequentialChain(
            chains=[wiki_chain],
            input_variables=["wiki_search"], 
            output_variables=["summary_list"],
            verbose=True
        )
        summary_wiki = summary_recchain({"wiki_search": articles_content})
        
        wiki_search = ["関連ワードを調査しました。"] + summary_wiki["summary_list"].split('\n')

        # return jsonify({'response': response['res'], 'wiki_search': wiki_search})
        
        counter += num_tokens_from_string(wiki_temp.format(wiki_search=articles_content), settings['model'])
        counter += num_tokens_from_string(summary_wiki["summary_list"], settings['model'])
    # Storageから検索を行うかどうかの判定
    if should_strage_search:
        # 埋め込みベクトルモデル
        embedding_model = settings['search_service_name']
        # Storageの名前
        container_name = settings['index_name']
        # Storage接続文字列
        connection_string = settings['strage_search_key']
        # BlobServiceClientを作成（Azure Storageへの接続クライアント）
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)

        # コンテナ内のBlobのリストを取得
        blob_list = blob_service_client.get_container_client(container_name).list_blobs()

        # テキストとそのコサイン類似度を格納するリストを初期化
        texts_and_similarities = []
        # OpenAIのAPIを使用してキーワードを埋め込みベクトルに変換
        response = openai.Embedding.create(input=message, model="text-embedding-ada-002")
        keyword_embedding = response['data'][0]['embedding']

        # コンテナ内の各Blobに対してループ処理
        for blob in blob_list:
            # Blobのクライアントを作成（特定のBlobにアクセスするためのクライアント）
            blob_client = blob_service_client.get_blob_client(container_name, blob.name)

            # Azure StorageからCSVファイルをダウンロード
            download_stream = blob_client.download_blob().readall()
            df = pd.read_csv(io.StringIO(download_stream.decode('utf-8')))

            # 'embedding'列をリストに戻す
            df['embedding'] = df['embedding'].apply(lambda x: list(map(float, x.strip('[]').split(', '))))

            # DataFrameの各行に対してループ処理
            for _, row in df.iterrows():
                # キーワードの埋め込みベクトルとテキストの埋め込みベクトルの間のコサイン類似度を計算
                cosine_sim = cs([keyword_embedding], [row['embedding']])[0][0]
                # COS類似度が0.8以上の場合のみ格納
                if cosine_sim > 0.8:
                    # テキスト、そのコサイン類似度、ページ番号、ファイル名をリストに追加
                    texts_and_similarities.append((cosine_sim, row['text'], row['page_num'], blob.name))

            # リストをコサイン類似度の降順にソート
            texts_and_similarities.sort(reverse=True)
            starage_str = []
            # キーワードと最もコサイン類似度が高いテキストのファイル名、ページ番号、テキストを出力
            range_sim_word = min(3, len(texts_and_similarities))
            for i in range(range_sim_word):
                # COS類似度が0.6以上の場合のみ格納
                if texts_and_similarities[i][0] > 0.6:
                    starage_str.append(f'File Name:{texts_and_similarities[i][3]}\nPage Number: {texts_and_similarities[i][2]}\n Text: {texts_and_similarities[i][1]}\n')
            # texts_and_similaritiesが空かどうかを確認
            if not starage_str:
                starage_str.append("類似文章が見当たりませんでした。")
            strage_search = ["Strageから関連文章を調べました。"] + starage_str
        
        counter += num_tokens_from_string("\n".join(starage_str), settings['model'])

    output_counter = str(counter)
    return jsonify({
        'wiki_search': wiki_search, 
        'bing_search': bing_search, 
        'rec_bing_search': rec_bing_search, 
        'recommendations': recommendations, 
        'strage_search': strage_search, 
        "output_counter": output_counter
    })

def get_bing_search_results_for_keywords(keywords, num_results=3, lang='ja-JP'):
    """
    与えられたキーワードのリストに対し、各キーワードについてBingで検索し、検索結果を取得する。

    Parameters
    ----------
    keywords : list of str
        検索するキーワードのリスト
    num_results : int, optional
        各キーワードに対して取得する検索結果の数 (default is 3)
    lang : str, optional
        使用する言語 (default is 'ja' for Japanese)

    Returns
    -------
    all_results : list of dict
        各キーワードについて取得した検索結果を含む辞書のリスト。
        各辞書はキーワード、検索結果のタイトル、URL、概要を含む。
    """
    subscription_key = settings['bing_search_v7_subscription_key']
    endpoint = settings['bing_search_v7_endpoint'].rstrip('/') + "/v7.0/search"
    all_results = []

    for keyword in keywords:
        params = {'q': keyword, 'count': num_results, 'mkt': lang}
        headers = {'Ocp-Apim-Subscription-Key': subscription_key}
        try:
            response = requests.get(endpoint, headers=headers, params=params)
            response.raise_for_status()

            json_response = response.json()
            results = []
            if 'webPages' in json_response:
                for item in json_response['webPages']['value']:
                    results.append({
                        'keyword': keyword,
                        'name': item['name'],
                        'url': item['url'],
                        'snippet': item['snippet']
                    })
            else:
                print(f"No web pages found for keyword {keyword}")
                
            all_results.extend(results)
        except Exception as ex:
            print(f"Error for keyword {keyword}: {str(ex)}")
    return all_results


def get_wikipedia_articles_for_keywords(keywords, num_articles=3, lang='ja'):
    """
    与えられたキーワードのリストに対し、各キーワードについてWikipedia記事を検索し、記事の情報を取得する。

    Parameters
    ----------
    keywords : list of str
        検索するキーワードのリスト
    num_articles : int, optional
        各キーワードに対して取得する記事の数 (default is 3)
    lang : str, optional
        使用する言語 (default is 'ja' for Japanese)

    Returns
    -------
    all_articles : list of dict
        各キーワードについて取得した記事の情報を含む辞書のリスト。
        各辞書はキーワード、タイトル、URL、記事の全文を含む。
    -------
    articles = get_wikipedia_articles_for_keywords(keywords)
    for article in articles:
        print('キーワード: ', article['keyword'])
        print('タイトル: ', article['title'])
        print('URL: ', article['url'])
        print('内容: ', article['content'])
        print('\n')
    """
    
    wikipedia.set_lang(lang)  # 言語を設定
    all_articles = []  # 全記事情報を保持するリスト

    for keyword in keywords:  # 各キーワードに対して
        try:
            titles = wikipedia.search(keyword, results=num_articles)  # キーワードでWikipediaを検索
            articles = []
            
            for title in titles:  # 取得した各タイトルに対して
                page = wikipedia.page(title)  # ページ情報を取得
                articles.append({  # 記事情報を辞書として追加
                    'keyword': keyword,  # 検索キーワード
                    'title': title,  # 記事のタイトル
                    'url': page.url,  # 記事のURL
                    'content': page.content  # 記事の全文
                })
            all_articles.extend(articles)  # 全記事情報リストに追加
        except wikipedia.DisambiguationError as e:  # 曖昧さ回避ページがヒットした場合のエラーハンドリング
            print(f"DisambiguationError for keyword {keyword}: {e.options}")  # エラーメッセージを出力
        
    return all_articles  # 全記事情報を返す

class MessageBlock:
    def __init__(self):
        self.messages = []
        self.outputs = []
        self.choices = []
        self.completions = []
        self.image = None
        self.models = []
        self.prompts = []
        self.tokens = 0

    def from_message(self, message):
        role = message["role"]
        if role == "system":
            content = message["content"]
            self.prompts.append(content["prompt"])
            self.completions.append(content["completion"])
            if "model" in content:
                self.models.append(content["model"])
            if "usage" in content:
                self.tokens += content["usage"]["total_tokens"]
            # Check if "choices" key exists in the content
            if "choices" in content:
                self.outputs.append(content["choices"][0]["message"]["content"])
                if content["choices"][0]["message"]["role"] == "image":
                    self.image = content["choices"][0]["message"]["content"]
                else:
                    self.choices.append(content["choices"][0]["message"]["content"])
        else:
            self.messages.append(message["content"])

    def update_from_message(self, message):
        self.from_message(message)
        if "models" in message:
            for model_message in message["models"]:
                self.from_message(model_message)
        if "system" in message:
            self.from_message(message["system"])

    def to_dict(self):
        return {
            "models": self.models,
            "prompts": self.prompts,
            "completions": self.completions,
            "tokens": self.tokens,
            "choices": self.choices,
            "outputs": self.outputs,
            "image": self.image
        }

    def end(self):
        self.completed = True
        return self.messages
class Interpreter:
    def __init__(self, max_tokens=150):
        self.messages = []
        self.initial_prompt = "You are a helpful assistant."
        self.max_tokens = max_tokens
        self.temperature = 0.7
        self.frequency_penalty = 0
        self.presence_penalty = 0.6
        self.stop_sequences = ["\n"]
        self.active_block = None
        self.completed = False

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})

    def chat(self, prompt_text):
        # Read the system message content from the file
        with open('./app/routes/system_message.txt', 'r') as file:
            system_content = file.read().strip()

        MAX_ATTEMPTS = 3
        full_response = ""
        cnt = 0
        for _ in range(MAX_ATTEMPTS):
            self.messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt_text}
            ]

            # API call simulation
            res = openai.ChatCompletion.create(
                model=settings['model'],
                messages=self.messages[-2:],  # Take the last two messages to avoid token limit
                temperature=settings['temperature']
            )
            
            response_content = res["choices"][0]["message"]["content"]
            cnt += res["usage"]["total_tokens"]
            full_response += response_content

            # Check if the response seems complete or if we need more information
            # Here, I'm checking if the response ends with a punctuation (indicative of completeness) 
            # but this can be adapted as needed.
            if response_content[-1] in ['.', '!', '?']:
                break
            else:
                # If not complete, use the previous response as the next prompt
                prompt_text = response_content

        return full_response, cnt

    def reset(self):
        self.messages = []
        self.active_block = None
        self.completed = False
