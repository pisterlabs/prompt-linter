import asyncio
import datetime
import re
import urllib
import urllib.parse
from contextlib import closing
from typing import List

import google.cloud.bigquery as bq
import google.cloud.bigquery.dbapi as bqapi
import httpx
import nltk
import spacy
import streamlit as st
from elasticsearch import Elasticsearch
from langchain.chat_models import ChatVertexAI, ChatOpenAI
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms import OpenAI, VertexAI
from langchain.output_parsers import NumberedListOutputParser
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, \
    ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage
from langchain.vectorstores.elasticsearch import ElasticsearchStore
from nltk import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from pstock import Bars

import summarizer.config as config
import summarizer.ner as ner
from summarizer.topic_sum import topic_aggregate_chain, aget_summaries, afind_top_topics, \
    RAGFusionRetriever, ArticleChunkHybridRetriever


class YFinance:
    """
    A temporary fix for Yahoo's main API being down.
    """

    user_agent_key = "User-Agent"
    user_agent_value = ("Mozilla/5.0 (Windows NT 6.1; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/58.0.3029.110 Safari/537.36")

    def __init__(self, ticker):
        self.yahoo_ticker = ticker

    def __str__(self):
        return self.yahoo_ticker

    async def _get_yahoo_cookie(self):
        headers = {self.user_agent_key: self.user_agent_value}
        client = httpx.AsyncClient()
        try:
            response = await client.get("https://fc.yahoo.com", headers=headers, follow_redirects=True)

            if not response.cookies:
                raise Exception("Failed to obtain Yahoo auth cookie.")

            return dict(response.cookies)
        finally:
            await client.aclose()

    async def _get_yahoo_crumb(self, cookie):
        headers = {self.user_agent_key: self.user_agent_value}
        client = httpx.AsyncClient()
        try:
            crumb_response = await client.get("https://query1.finance.yahoo.com/v1/test/getcrumb",
                                              headers=headers, follow_redirects=True,
                                              cookies=cookie)
            crumb = crumb_response.text

            if crumb is None:
                raise Exception("Failed to retrieve Yahoo crumb.")
            return crumb
        finally:
            await client.aclose()

    async def aget_bars(self, period):
        interval_map = {
            "1d": "30m",
            "5d": "1h",
            "1mo": "1d",
            "3mo": "1d",
            "6mo": "1d"
        }

        temp = await Bars.get(self.yahoo_ticker, period=period, interval=interval_map[period])
        return temp.df

    async def aget_info(self):
        # Yahoo modules doc informations :
        # https://cryptocointracker.com/yahoo-finance/yahoo-finance-api
        cookie = await self._get_yahoo_cookie()
        crumb = await self._get_yahoo_crumb(cookie)
        ret = {}

        headers = {self.user_agent_key: self.user_agent_value}

        yahoo_modules = ("financialData,"
                         "quoteType,"
                         "defaultKeyStatistics,"
                         "assetProfile,"
                         "summaryDetail")

        url = ("https://query1.finance.yahoo.com/v10/finance/"
               f"quoteSummary/{self.yahoo_ticker}"
               f"?modules={urllib.parse.quote_plus(yahoo_modules)}"
               f"&ssl=true&crumb={urllib.parse.quote_plus(crumb)}")

        client = httpx.AsyncClient()
        try:
            info_response = await client.get(url, headers=headers, follow_redirects=True, cookies=cookie)
            info = info_response.json()
            if not info["quoteSummary"] or not info["quoteSummary"]["result"]:
                raise FileNotFoundError(self.yahoo_ticker)
            info = info['quoteSummary']['result'][0]

            for mainKeys in info.keys():
                for key in info[mainKeys].keys():
                    if isinstance(info[mainKeys][key], dict):
                        try:
                            ret[key] = info[mainKeys][key]['raw']
                        except (KeyError, TypeError):
                            pass
                    else:
                        ret[key] = info[mainKeys][key]

            return ret
        finally:
            await client.aclose()

    @property
    def info(self):
        return asyncio.run(self.aget_info())

    def __hash__(self):
        return hash(self.yahoo_ticker)


async def fetch_yahoo_kpi(ticker_symbol):
    ticker = YFinance(ticker_symbol)
    info = await ticker.aget_info()
    market_cap = info["marketCap"]
    if not market_cap:
        raise FileNotFoundError(ticker_symbol)
    return info


async def fetch_yahoo_data(ticker, period):
    ticker = YFinance(ticker)
    return await ticker.aget_bars(period)


async def fetch_ticker(ticker_symbol, query, period):
    print(f"Fetching Ticker: {ticker_symbol}")
    info, data = await asyncio.gather(fetch_yahoo_kpi(ticker_symbol), fetch_yahoo_data(ticker_symbol, period))
    summary = info.get('longBusinessSummary', 'No summary available.')
    print(f"Found summary: {summary}")
    kpis = ner.extract_relevant_field(query, summary, info)
    print(f"Found KPI: {kpis}")
    result = {}
    for g in kpis.groups:
        if g.group_title not in result:
            result[g.group_title] = {}
        for m in g.group_members:
            print("Trying to get KPI: " + m)
            if m in info:
                print("Adding KPI: " + m)
                result[g.group_title][m] = info[m]

    return ticker_symbol, data, summary, result


async def fetch_all_tickers(tickers, query, period):
    coros = [fetch_ticker(t, query, period) for t in tickers]
    results = await asyncio.gather(*coros, return_exceptions=True)
    return [r for r in results if not isinstance(r, Exception)]


@st.cache_data
def get_es_credentials():
    with open(config.ES_KEY_PATH, "r") as fp:
        es_key = fp.read().strip()
    with open(config.ES_CLOUD_ID_PATH, "r") as fp:
        es_id = fp.read().strip()
    return es_id, es_key


@st.cache_data
def get_openai_credentials():
    with open(config.OPENAI_API) as fp:
        key = fp.read().strip()
    return key


@st.cache_resource
def get_async_elastic():
    es_id, es_key = get_es_credentials()
    return Elasticsearch(cloud_id=es_id, api_key=es_key)


@st.cache_resource
def get_embeddings():
    embedding = SentenceTransformerEmbeddings(model_name=config.FILTER_EMBEDDINGS, model_kwargs={'device': 'cpu'})
    return embedding


@st.cache_resource
def get_keyword_extractor(model="en_core_web_sm"):
    nlp = spacy.load(model, enable=["ner"])
    return nlp


@st.cache_resource
def get_topic_store():
    embedding = get_embeddings()
    es_id, es_key = get_es_credentials()
    topic_store = ElasticsearchStore(index_name=config.ES_TOPIC_INDEX, embedding=embedding,
                                     es_cloud_id=es_id, es_api_key=es_key,
                                     vector_query_field=config.ES_TOPIC_VECTOR_FIELD,
                                     query_field=config.ES_TOPIC_FIELD)
    return topic_store


@st.cache_resource
def get_article_store():
    embedding = get_embeddings()
    es_id, es_key = get_es_credentials()
    article_store = ElasticsearchStore(index_name=config.ES_ARTICLE_INDEX, embedding=embedding,
                                       es_cloud_id=es_id, es_api_key=es_key,
                                       vector_query_field=config.ES_ARTICLE_VECTOR_FIELD,
                                       query_field=config.ES_ARTICLE_FIELD)
    return article_store


@st.cache_data(ttl="7d")
def get_model_num():
    """
    Gets the latest servable topic model

    :returns: the latest servable model number
    """

    query = "SELECT id FROM Articles.TopicModel WHERE servable ORDER BY fit_date DESC LIMIT 1"
    with bq.Client(project=config.GCP_PROJECT) as client:
        with closing(bqapi.Connection(client=client)) as connection:
            with closing(connection.cursor()) as cursor:
                cursor.execute(query)
                return cursor.fetchone()[0]


@st.cache_resource
def setup_nltk():
    nltk.download("punkt")
    nltk.download('wordnet')
    nltk.download('stopwords')
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    stop_words = set([w.lower() for w in stopwords.words("english")])
    return stemmer, lemmatizer, {s.lower() for s in stop_words}


def tex_escape(text):
    """
        :param text: a plain text message
        :return: the message escaped to appear correctly in LaTeX
    """
    return text.replace("$", "＄")


def escape_markdown(text: str, version: int = 1, entity_type: str = None) -> str:
    """
    Helper function to escape telegram markup symbols.

    Args:
        text (:obj:`str`): The text.
        version (:obj:`int` | :obj:`str`): Use to specify the version of telegrams Markdown.
            Either ``1`` or ``2``. Defaults to ``1``.
        entity_type (:obj:`str`, optional): For the entity types ``PRE``, ``CODE`` and the link
            part of ``TEXT_LINKS``, only certain characters need to be escaped in ``MarkdownV2``.
            See the official API documentation for details. Only valid in combination with
            ``version=2``, will be ignored else.
    """
    if int(version) == 1:
        escape_chars = r'_*`['
    elif int(version) == 2:
        if entity_type in ['pre', 'code']:
            escape_chars = r'\`'
        elif entity_type == 'text_link':
            escape_chars = r'\)'
        else:
            escape_chars = r'_*[]()~`>#+-=|{}.!'
    else:
        raise ValueError('Markdown version must be either 1 or 2!')

    return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)


def get_qa_llm(kind=config.QA_MODEL, max_token=256):
    """
    Gets a langchain LLM for QA.

    :param kind: the type of model. Currently it supports PaLM2 ("vertexai"), GPT-3.5 ("openai").
    :param max_token: the maximum number of tokens that can be emitted.
    """

    if kind == "vertexai":
        plan_llm = VertexAI(
            project=config.GCP_PROJECT,
            temperature=0,
            model_name="text-bison",
            max_output_tokens=max_token
        )
        return plan_llm
    elif kind == "openai":
        plan_llm = OpenAI(model_name="gpt-3.5-turbo-instruct", openai_api_key=get_openai_credentials(),
                          temperature=0, max_tokens=max_token)
        return plan_llm
    elif kind == "custom":
        plan_llm = OpenAI(openai_api_base=config.QA_API_SERVER, model_name=config.QA_API_MODEL,
                          max_tokens=max_token, temperature=0,
                          openai_api_key="EMPTY", presence_penalty=1)
        return plan_llm
    else:
        raise NotImplemented()


def get_summary_llm(kind=config.SUM_MODEL, max_token=256):
    """
    Gets a langchain LLM for key-point summarization.

    :param kind: the type of model. Currently it supports PaLM2 ("vertexai"), GPT-3.5 ("openai"),
    and Mistral-7B ("custom")
    :param max_token: the maximum number of tokens that can be emitted.
    """

    if kind == "vertexai":
        plan_llm = VertexAI(
            project=config.GCP_PROJECT,
            temperature=0,
            model_name="text-bison",
            max_output_tokens=max_token
        )
        return plan_llm
    elif kind == "custom":
        plan_llm = OpenAI(openai_api_base=config.SUM_API_SERVER, model_name=config.SUM_API_MODEL,
                          max_tokens=max_token, temperature=0,
                          openai_api_key="EMPTY", presence_penalty=1)
        return plan_llm
    elif kind == "openai":
        plan_llm = OpenAI(model_name="gpt-3.5-turbo-instruct", openai_api_key=get_openai_credentials(),
                          temperature=0, max_tokens=max_token)
        return plan_llm
    else:
        raise NotImplemented()


def get_rewrite_llm(kind=config.REWRITE_MODEL, max_token=1024):
    """
    Gets a langchain LLM for injecting context into query.

    :param kind: the type of model. Currently it supports PaLM2 ("vertexai"), GPT-3.5 ("openai")
    :param max_token: the maximum number of tokens that can be emitted.
    """
    if kind == "vertexai":
        plan_llm = ChatVertexAI(
            project=config.GCP_PROJECT,
            temperature=0,
            model_name="chat-bison",
            max_output_tokens=max_token
        )
        return plan_llm
    elif kind == "openai":
        plan_llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=get_openai_credentials(),
                              temperature=0, max_tokens=max_token)
        return plan_llm
    else:
        raise NotImplemented()


async def answer_question(query, now, delta, model, augmented_queries):
    """
    Creates an async task to answer the user's query for a set of topics

    :param query: the user query
    :param augmented_queries: rag fusion augmented queries
    :param now: the current time
    :param delta: the timedelta for how far back to go
    :param model: the topic model number
    """

    plan_llm = get_qa_llm()
    retriever = ArticleChunkHybridRetriever(
        elastic_client=get_async_elastic(),
        spacy_nlp=get_keyword_extractor(),
        encoder=get_embeddings(),
        fetch_k=config.FUSION_SRC_CHUNKS,
        candidate_k=config.FUSION_SRC_CHUNKS * 3,
        chunk_k=config.FUSION_SRC_CHUNKS,
        time_delta=delta,
        now=now,
        model=model
    )
    retriever = RAGFusionRetriever(rewrite_llm=get_rewrite_llm(),
                                   base_retriever=retriever, queries=[query] + list(augmented_queries))
    qa_agg_chain = topic_aggregate_chain(plan_llm, retriever)
    qa_result = await qa_agg_chain.acall(query)
    return {
        "answer": qa_result["result"].replace("\n\n", "\n").strip(),
        "sources": qa_result["source_documents"]
    }


def determine_relevant_keypoints(query: str, key_points: List[str]):
    stem, lemma, stop = setup_nltk()
    query_words = nltk.word_tokenize(query)
    query_words = [w.lower() for w in query_words if w not in stop and w.isalpha()]
    query_words = {stem.stem(lemma.lemmatize(w)) for w in query_words}
    key_points_tokens = set()
    for s in key_points:
        key_words = nltk.word_tokenize(s)
        key_words = [w.lower() for w in key_words if w not in stop and w.isalpha()]
        key_words = {stem.stem(lemma.lemmatize(w)) for w in key_words}
        key_points_tokens.update(key_words)
    jaccard = len(query_words.intersection(key_points_tokens)) / len(query_words.union(key_points_tokens))
    return jaccard


async def find_summaries(query, topics, model, now, delta):
    """
    Creates an async corotine for generating the key point summaries.

    :param query: the query string from the user
    :param topics: the relevant topics
    :param model: the topic model number
    :param now: the current time
    :param delta: the timedelta for how far back to go
    """

    plan_llm = get_summary_llm()
    summaries = await aget_summaries(query, topics, now, delta, model, get_article_store(), plan_llm)
    return summaries


async def get_qa_result(query, now, delta, model, queries):
    """
    async function for concurrently generating the qa response, and also the key-point summaries

    :param query: the user query string
    :param queries: the fusion rag augmented queries
    :param now: the current time
    :param delta: the timedelta for how far back to go
    :param model: the topic model number
    :returns: A dictionary in the form of
    {"qa": qa answer, "summaries": [{"title": tagline, "keypoints": [keypoints 1, kp 2,...]}}, ...]
    """

    topic_results = await afind_top_topics(get_topic_store(), query, now, delta, model, k=config.TOPIC_SUM_K)
    topic_nums = [topic.metadata["topic"] for topic in topic_results]
    qa_coro = asyncio.ensure_future(answer_question(query, now, delta, model, queries))
    sum_coro = asyncio.ensure_future(find_summaries(query, topic_nums, model, now, delta))

    qa_completed, summaries = await asyncio.gather(qa_coro, sum_coro)
    filtered_sums = []
    for s in summaries:
        distance = determine_relevant_keypoints(query + ". " + qa_completed["answer"], [s["title"]] + s["keypoints"][:3])
        print(f"Found distance {distance} for summary {s['title']}")
        if distance >= 0.03:
            filtered_sums.append(s)

    return {
        "qa": qa_completed,
        "summaries": filtered_sums
    }


def get_fusion_chain(history, k=3):
    output_parser = NumberedListOutputParser()
    rewrite_system = SystemMessagePromptTemplate.from_template(config.REWRITE_SYSTEM_PROMPT)
    rewrite_user = HumanMessagePromptTemplate.from_template(config.REWRITE_USER_PROMPT)
    rewrite_history = []
    for h in history[-k:]:
        rewrite_history.append(HumanMessage(content=h["user_text"]))
        rewrite_history.append(AIMessage(content=h["resp_text"]))
    rewrite_chat = ChatPromptTemplate.from_messages([rewrite_system, *rewrite_history, rewrite_user])
    fusion_system = SystemMessagePromptTemplate.from_template(config.FUSION_SYSTEM_PROMPT)
    fusion_user = HumanMessagePromptTemplate.from_template(config.FUSION_USER_PROMPT)
    fusion_chat = ChatPromptTemplate.from_messages([fusion_system, fusion_user])
    llm = get_rewrite_llm()
    return rewrite_chat | llm, fusion_chat.partial(
        format_instructions=output_parser.get_format_instructions()) | llm | output_parser


async def augment_query(text, history):
    queries = []
    try:
        rewrite_chain, fusion_chain = get_fusion_chain(history)
        if len(history) > 0:
            rewritten_query = await rewrite_chain.ainvoke({"query": text})
        else:
            rewritten_query = AIMessage(content=text)
        augmented_queries = await fusion_chain.ainvoke({"query": rewritten_query.content})
        queries.append(rewritten_query.content)
        queries.extend(augmented_queries)
    except Exception as e:
        print(f"Failed to apply fusion rag: {e}")
        queries.append(text)
    return queries


period_map = {
    "1d": datetime.timedelta(days=1),
    "5d": datetime.timedelta(days=5),
    "1mo": datetime.timedelta(days=30),
    "3mo": datetime.timedelta(days=91),
    "6mo": datetime.timedelta(days=183)
}


def finbot_response(text, period, history):
    """
    Gets the qa and summarization result given query and timeframe:

    :param text: the user query
    :param period: string representing how far back to go
    """

    now = datetime.datetime(year=2023, month=11, day=10)
    delta = period_map[period]
    topic_model = get_model_num()
    actual_history = history[:-1]
    augmented_queries = asyncio.run(augment_query(text, actual_history))
    print(augmented_queries)
    return asyncio.run(get_qa_result(augmented_queries[0], now, delta, topic_model, augmented_queries[1:]))


if __name__ == "__main__":
    ticker = YFinance("AAPL")
    print(ticker.info)
