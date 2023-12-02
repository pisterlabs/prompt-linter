from langchain.text_splitter import TokenTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from bs4 import BeautifulSoup
from duckduckgo_search import ddg
import openai
#import pycreds
import logging
import threading
import queue
from readability import Document
import scrapy
from scrapy.crawler import CrawlerProcess
import os
import json
from dotenv import load_dotenv
from .chat import *
from .counter import token_counter


from .chat import *
from .prompt import *
from .vectorstore import VectorStore


# 定数定義
CONFIG = "config.ini"
# .envファイルからAPIキーを読み込む
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Embeddingモデル
EMBEDDING_MODEL = "text-embedding-ada-002"


def get_urls_messages(chapters_reply, chapter, json_format):
    triggering_prompt = TRIGGERING_PROMPT.format(chapters_reply)
    triggering_prompt = create_chat_message(SYSTEM_ROLE, triggering_prompt)
    instruction = f"""
    {chapter}\n
    上記の章の問題を作成する際に役立つウェブサイトのURLを出力しなさい。
    """
    input_prompt = create_chat_message(USER_ROLE, instruction)
    format = f"""
    出力形式については以下のJSON形式のみ認められております。\n
    {json_format}\n
    必ずpythonのjson.loadsで読み込めるような形式に合致させてください。
    """
    system_prompt = create_chat_message(SYSTEM_ROLE, format)
    return [triggering_prompt, input_prompt, system_prompt]

def get_useful_urls(chapters_reply, chapter):
    """
    説明：各章に関連する有用なウェブサイトのURLを取得
    """
    json_format = {
        "chapter": f"{chapter}", # MEMO プロパティは「"」で囲まれている必要がある
        "urls": ["url1", "url2", "url3"]
    }
    messages = get_urls_messages(chapters_reply, chapter, json_format)
    response = create_chat_completion(
        messages,
        temperature=0.1,
        api_key=OPENAI_API_KEY
    )
    return response


def create_summary_from_single_chunk(url, title, text, idx, q=None):
    """
    説明：OpenAIのLLMによって有益なテキストデータのみを取り出す
    引数：url, title, ウェブサイトのテキスト
    戻り値：抽出されたテキストデータ
    """
    token_limit = 8000
    logger = logging.getLogger('ddgsearch')
    logger.info(f"extracting useful information from chunk {idx}, title: {title}")

    messages = []
    messages.append(create_message(text, url, title))
    current_token_used = token_counter(messages)
    num_tokens = token_limit - current_token_used

    
    openai.api_key = os.getenv('OPENAI_API_KEY')

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        max_tokens=num_tokens,
        temperature=0.2
    )

    if q:
        # キューに結果を入れる（GPTからの出力をキューに入れる）
        q.put((idx, response.choices[0].message['content']))
    logger.info(f"Done extracting useful information from chunk {idx}, title: {title}")

    text = response.choices[0].message['content']

    lines = text.splitlines()
    if "useful information" in lines[0].lower():
        text = "\n".join(lines[1:])

    return (idx, text)


def create_summary(url, title, text, max_chunks):
    """
    説明：長いテキストをチャンクに分割して処理
    """
    chunks = [text[i*3000: i*3000+3100] for i in range(len(text)//3000)]
    chunks = chunks[:max_chunks]

    threads = []

    q = queue.Queue()

    for idx, chunk in enumerate(chunks):
        # 並列処理化
        t = threading.Thread(target=create_summary_from_single_chunk, args=(url, title, chunk, idx, q))
        threads.append(t)
        t.start()

    # すべての処理が終了
    for t in threads:
        t.join()

    results = []
    while not q.empty():
        results.append(q.get())

    logger = logging.getLogger('ddgsearch')
    logger.info(f"Got {len(results)} results from the queue")

    # ソート処理
    results.sort(key=lambda x: x[0])
    
    # 分割されたテキストを結合
    text = ''.join([x[1] for x in results])

    return text

def extract_useful_information_from_single_chunk(url, title, text, idx, q=None):
    """
    説明：OpenAIのLLMによって有益なテキストデータのみを取り出す
    引数：url, title, ウェブサイトのテキスト
    戻り値：抽出されたテキストデータ
    """
    logger = logging.getLogger('ddgsearch')
    logger.info(f"extracting useful information from chunk {idx}, title: {title}")

    prompt = f"""
    これはURL: {url}\n
    これはタイトル: {title}\n
    これはbs4.BeautifulSoupによって抽出されたウェブページの一部のテキストデータです。\n
    ----------\n
    {text}\n
    ----------\n
    \n
    ウェブページには多くの無駄なゴミが含まれています。例えば、多くの広告、多くのリンク集や
    そのウェブページのトピックには関係のないテキストなどがあります。私達はもとのテキストデータから
    有益な情報のみを抽出したいのです。\n
    \n
    あなたはURLやタイトルをテキストデータのコンテクストを理解するために使うことができます。
    テキストデータから有益な情報のみを抽出してください。しかし、もとのテキストデータの内容を書き換えることなく、有益な情報のみを抽出してください。
    """

    openai.api_key = os.getenv('OPENAI_API_KEY')

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=1000,
        temperature=0.2,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    if q:
        q.put((idx, response['choices'][0]['text']))
    logger.info(f"Done extracting useful information from chunk {idx}, title: {title}")

    text = response['choices'][0]['text']

    lines = text.splitlines()
    if "useful information" in lines[0].lower():
        text = "\n".join(lines[1:])

    return (idx, text)


def extract_useful_information(url, title, text, max_chunks):
    """
    説明：長いテキストをチャンクに分割して処理
    """
    chunks = [text[i*1000: i*1000+1100] for i in range(len(text)//1000)]
    chunks = chunks[:max_chunks]

    threads = []

    q = queue.Queue()

    for idx, chunk in enumerate(chunks):
        # 並列処理化
        t = threading.Thread(target=extract_useful_information_from_single_chunk, args=(url, title, chunk, idx, q))
        threads.append(t)
        t.start()

    # すべての処理が終了
    for t in threads:
        t.join()

    results = []
    while not q.empty():
        results.append(q.get())

    logger = logging.getLogger('ddgsearch')
    logger.info(f"Got {len(results)} results from the queue")

    # ソート処理
    results.sort(key=lambda x: x[0])
    
    # 分割されたテキストを結合
    text = ''.join([x[1] for x in results])

    return text


def readability(input_text):
    doc = Document(input_text)

    summary = doc.summary()

    soup = BeautifulSoup(summary, 'html.parser')
    summary_text = soup.get_text()

    return summary_text


def remove_duplicate_empty_lines(input_text):
    """
    説明：不要な重複の空白行を削除
    """
    lines = input_text.splitlines()

    fixed_lines = []
    for index, line in enumerate(lines):
        if line.strip() == '':
            if index != 0 and lines[index-1].strip() != '':
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)

    return '\n'.join(fixed_lines)

class MySpider(scrapy.Spider):
    """
    説明：crawler用のspiderクラス
    """
    name = 'myspider'
    start_urls = None
    clean_with_llm = False
    results = []

    def __init__(self, start_urls, clean_with_llm, *args, **kwargs):
        super(MySpider, self).__init__(*args, **kwargs)
        self.start_urls = start_urls
        self.clean_with_llm = clean_with_llm

    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(url, callback=self.parse)

    def parse(self, response):
        logger = logging.getLogger('ddgsearch')
        logger.info(f"***Parsing {response.url}...")

        body_html = response.body.decode('utf-8')

        url = response.url

        soup = BeautifulSoup(body_html, 'html.parser')
        title = soup.title.string
        text = soup.get_text()
        text = remove_duplicate_empty_lines(text)

        if self.clean_with_llm:
            useful_text = create_summary(url, title, text, 50)
            # useful_text = extract_useful_information(url, title, text, 50)
        else:
            useful_text = readability(body_html)
        useful_text = remove_duplicate_empty_lines(useful_text)

        self.results.append({
            'url': url,
            'title': title,
            'text': text,
            'useful_text': useful_text
        })


def ddgsearch(query, num_results=10, clean_with_llm=False):
    """
    説明：duckduckgoによってウェブサーチを行う
    引数：検索用のクエリ（必須）
    戻り値：検索結果
    """
    results = ddg(query, max_results=num_results)

    # ログの出力
    logger = logging.getLogger('ddgsearch')
    logger.info(f"Got {len(results)} results from the search.")
    logger.debug(f"Results: {results}")

    # urlの抽出
    urls = [result['href'] for result in results]
    urls = urls[:num_results]

    process = CrawlerProcess()
    process.crawl(MySpider, urls, clean_with_llm)
    process.start()

    return MySpider.results


def gpt_search(chapters_reply, chapter, clean_with_llm=False):
    """
    説明：ウェブサーチを行い、要約を作成
    引数：検索用のクエリ（必須）
    戻り値：検索結果
    """
    logger = logging.getLogger('ddgsearch')
    # 検索結果の取得
    results = get_useful_urls(chapters_reply, chapter)
    data = json.loads(results)
    urls = data['urls']
    # 取得したURLの数を確認
    logger.info(f"Got {len(urls)} results from the search.")
    # URL確認
    logger.info("URLs: {}".format(urls))

    process = CrawlerProcess()
    process.crawl(MySpider, urls, clean_with_llm)
    process.start()

    return MySpider.results

def create_message(chunk, url, title):
    """
    説明：ウェブページ要約
    引数：ウェブページの一部、指示
    戻り値：要約用のメッセージ
    """
    prompt = WEB_SUMMARY_PROMPT.format(chunk, url, title)

    return create_chat_message(USER_ROLE, prompt)


def get_urls(query, num_results=5):
    """
    引数：検索ワード、取得サイト数
    """
    results = ddg(query, max_results=num_results)

    # ログの出力
    logger = logging.getLogger('ddgsearch')
    logger.info(f"Got {len(results)} results from the search.")
    logger.debug(f"Results: {results}")

    # urlの抽出
    urls = [result['href'] for result in results]
    urls = urls[:num_results]

    return urls


def split_texts(urls, chunksize=1000):
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)

    return texts

def web_search(search_words, query, n_results=1):
    """
    説明：ウェブ検索を行い最も関連性の高い情報を抽出
    """
    texts = split_texts(get_urls(search_words))
    substitution = {}
    vector_store = VectorStore(OPENAI_API_KEY, substitution, CONFIG)
    # ベクトルDB作成
    vector_store.create_db(texts)

    # クエリをEmbedding化
    query_embed = openai.Embedding.create(
        input=query,
        model=EMBEDDING_MODEL
    )["data"][0]["embedding"]

    query_results = vector_store.collection.query(
        query_embeddings=query_embed,
        n_results=n_results,
        include=["documents"]
    )

    return query_results["documents"][0]
