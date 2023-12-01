import langchain
import pandas as pd
import numpy as np
import openai
import os

class APIkeyImport:
    def __init__(self, Openai_api_key=None, Youtube_api_key=None):
        self.OPENAI_API_KEY = Openai_api_key
        self.YOUTUBE_API_KEY = Youtube_api_key
    
    def byEnv(self):
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import CommaSeparatedListOutputParser

class langchainModule(APIkeyImport): # 검색어 뽑아주는 모듈(langchain), 뉴스 기사 본문 요약하는 모듈(langchain), 뉴스 기사와 댓글의 연관성 분석(langchain)
    def __init__(self):
        super().__init__()
        self.strict_model = ChatOpenAI(temperature=0, model_name='gpt-4', api_key=self.OPENAI_API_KEY)
        self.smooth_model = ChatOpenAI(temperature=0.5, model_name='gpt-4', api_key=self.OPENAI_API_KEY)
        self.list_parser = CommaSeparatedListOutputParser()
        self.templates = {
            "search":
            ChatPromptTemplate.from_messages(
                [
                    ("system", """You are a keyword generator that generates search keywords from news articles.
                     You are given a title, description, and content of the news article.
                     You should generate search keywords that can be used to search for the news article directly and strongly related to the summarization.
                     """),
                    ("human", "title: {title}, description: {description}, content: {content}"),
                ]
            ),
            "summarization": 
            ChatPromptTemplate.from_messages(
                [
                    ("system", """You are a summarizer that summarizes news articles.
                     You should summarize given news article.
                     """),
                    ("human", "title: {title}, description: {description}, content: {content}"),
                ]
            ),
            "relation": 
            ChatPromptTemplate.from_messages(
                [
                    ("system", """You should analyze the relation between news articles and comments.
                     You are given a content of a news article and comments.
                     You should write reports of the relation between the news article and the comments whether it is related or not, and how much it is related.
                     """
                     ),
                    ("human", "content: {content}, comments: {comments}"),
                ]
            ),
            "topic seperation": 
            ChatPromptTemplate.from_messages(
                [
                    ("system", """You should seperate news article of query. Query is kind of a set of a news article which is not seperated.
                     You are given a title, description, content of a news.
                     One query can contain several articles. But some case, news can contain only one article. If then, never mind seperating it. Just return the original content of the news as a return type where written downside of this instruction.
                     
                     Range of a topic is one article. If the content of the news are connected by meaning, you NEVER seperate it by topic.
                     You should seperate the content in original query by article, with list format consisted of article composing original content.
                     Some case, trash datas such as advertisement, non-news contents can be included in the news.
                     If you find trash datas, you should remove it.
                     ex) [article1, article2, article3]
                     """
                     ),
                    ("human", "title: {title}, description: {description}, content: {content}"),
                ]
            ),
            "report":
            ChatPromptTemplate.from_messages(
                [
                    ("system", """You should report a overall result of the news article in Korean.
                        You are given a title, description, content, analysis of the comments, and relation between comments and article of a news.
                        You should write a report of the news article.
                        The report can contain the following contents, and your overall analysis would be about the inclination of the news article,
                        how comments interact with article, and how much the article is related to the comments in fact.
                        And also you should give an insight of the inclination, aggression of the news comments by given query.
                     
                        You can write the report in Korean.
                        You should write the report in markdown format.
                        Output format: MAKRDOWN
                     """
                     ),
                    ("human", "title: {title}, description: {description}, content: {content}, comments analytics: {comments_analytics}, relation: {relation}"),
                ]
            ),
        }
    
    def search_keyword_generate(self, title, description, content):
        messages = self.templates["search"]
        self.search_keyword_generate_chain = messages | self.smooth_model
        return self.search_keyword_generate_chain, self.search_keyword_generate_chain.invoke({"title": title, "description": description, "content": content})

    def summarize_content(self, title, description, content):
        messages = self.templates["content"]
        self.summarize_content_chain = messages | self.strict_model
        return self.summarize_content_chain, self.summarize_content_chain.invoke({"title": title, "description": description, "content": content})

    def calculate_relation(self, content, comments):
        messages = self.templates["relation"]
        self.calculate_relation_chain = messages | self.smooth_model
        return self.calculate_relation_chain, self.calculate_relation_chain.invoke({"content": content, "comments": str(comments)})
    
    def topic_seperation(self, title, description, content):
        content = content.replace(",", " ")
        """
        TODO: Slow speed because of generation of gpt-4,
        try to solve this problem with classical way, instead use openai api directly
        """
        messages = self.templates["topic seperation"]
        self.topic_seperation_chain = messages | self.strict_model
        output = self.topic_seperation_chain.invoke({"title": title, "description": description, "content": content}).content
        return self.topic_seperation_chain, self.list_parser.parse(output)

    def report(self, title, description, content, comments_analytics):
        relation = self.calculate_relation(content, comments_analytics)
        messages = self.templates["report"]
        self.report_chain = messages | self.smooth_model
        return self.report_chain, self.report_chain.invoke({"title": title, "description": description, "content": content, "comments_analytics": comments_analytics, "relation": relation})
    
import requests
from urllib import parse
from googleapiclient import discovery
from datetime import datetime, timedelta, timezone
from pytube import YouTube

class CollectModule(langchainModule, APIkeyImport):
    def __init__(self):
        super().__init__()
        self.youtube = discovery.build('youtube', 'v3', developerKey=APIkeyImport.YOUTUBE_API_KEY)
        openai.api_key = APIkeyImport.OPENAI_API_KEY
    
    def search_keyword_generate(self, title, description, content, is_seperate=False):
        if is_seperate == False:
            return [super().search_keyword_generate(title, description, content)[1]]
        else:
            list_of_topic = self.topic_seperation(title, description, content)
            list_of_keywords = []
            for topic in list_of_topic:
                list_of_keywords.append(super().search_keyword_generate(title, description, topic)[1])
            return list_of_keywords

    def topic_seperation(self, title, description, content):
        return super().topic_seperation(title, description, content)[1]

    def youtube_search_one_day(self, keyword, dayBefore=0) -> dict: # dayBefore는 n일 전부터 n+1일 전까지 수집 가능하도록 해줌
        now = datetime.now()
        dateEnd = (now - timedelta(days=dayBefore)).isoformat() # 현재 시점
        dateStart = (now - timedelta(days=(dayBefore+1))).isoformat() # 현재 시점으로부터 24시간 전

        req = self.youtube.search().list(q=keyword, part='snippet',
                                        videoDuration = 'medium',
                                        order='viewCount',
                                        type='video',
                                        regionCode='KR', #한국 영상만 검색
                                        videoCategoryId=25,
                                        maxResults=3,
                                        publishedAfter = dateStart+'Z',
                                        publishedBefore = dateEnd+'Z',
                                    )
        res = req.execute()
        return res

    def youtube_search_multi_day(self, query, length=3) -> dict: # youtube_search_one_day의 dayBefore 인자를 바꿔가며 n일치 뉴스를 수집 가능
        result_dict = {}

        for i in range(length):
            result = self.youtube_search_one_day(query, i)
            result_dict[i] = result

        return result_dict

    def get_comments_by_id(self, videoID, maxresult) -> list:
        
        req = self.youtube.commentThreads().list(
            part='snippet',
            videoId = videoID,
            maxResults = maxresult
        )
        res = req.execute()
        
        comment_list = []
        for n in range(len(res['items'])):
            comment_list.append(res['items'][n]['snippet']['topLevelComment']['snippet']['textOriginal'])

        return comment_list

    def speech_to_text(self, videoID):
        try:
            yt = YouTube(f'https://youtu.be/{videoID}')
            file_path = yt.streams.filter(only_audio=True).first().download(output_path='/data/audio', filename=f'{videoID}.wav')
            audio_file = open(f'/data/audio/{videoID}.wav', 'rb')
            transcript = openai.audio.transcriptions.create(model='whisper-1', file=audio_file)
        except:
            return np.nan
        return transcript

    def parse_response(self, videores_dict):
        result_dict = {}

        for i in range(len(videores_dict)):
            videores = videores_dict[i]
            for n in range(len(videores['items'])):
                result_dict[i] = {"title": videores['items'][n]['snippet']['title'],
                                "content": self.speech_to_text(videores['items'][n]['id']['videoId']),
                                "publishedAt": videores['items'][n]['snippet']['publishedAt'],
                                "publisher": videores['items'][n]['snippet']['channelTitle'],
                                "description": videores['items'][n]['snippet']['description'],
                                "source": 1,
                                "comment": self.get_comments_by_id(videores['items'][n]['id']['videoId'], 50),
                                "code": videores['items'][n]['id']['videoId']}

        return result_dict

    def youtube_collect(self, keyword_list):
        result_dict_list = []

        for keyword in keyword_list:
            videores_dict = self.youtube_search_multi_day(keyword)
            result_dict = self.parse_response(videores_dict)
            result_dict_list.append(result_dict)

        return result_dict

    #TODO: 네이버 뉴스 수집
    def naver_collect(self, keyword_list):
        return None

    def news_collect(self, title, description, content):
        keyword_list = self.search_keyword_generate(title, description, content)
        youtube_result = self.youtube_collect(keyword_list)
        naver_result = self.naver_collect(keyword_list)
        
        youtube_result_df = pd.DataFrame(youtube_result)
        naver_result_df = pd.DataFrame(naver_result)
        total_result = pd.concat([youtube_result_df, naver_result_df], axis=1).T
        return total_result
    
class ClassifierSet(APIkeyImport):
  def __init__(self):
    super().__init__()
    from openai import OpenAI
    self.client = OpenAI(api_key=self.OPENAI_API_KEY)
    self.cate_model_id = "ft:gpt-3.5-turbo-1106:team-honeybread::8LLRyOr7"
    self.bias_model_id = "ft:gpt-3.5-turbo-1106:team-honeybread::8KwjwyJF"
    self.aggr_model_id = "ft:gpt-3.5-turbo-1106:team-honeybread::8KyLe6t9"

  def cate_classifier(self, query):
    completion = self.client.chat.completions.create(
    model=self.cate_model_id,
    messages=[
        {"role": "system", "content": """You are a great data classifier conducting on a online comment at news article.
        All you have to do is convert the comments in the query into category about the comments."""},
        {"role": "user", "content": query},
      ]
    )
    return completion.choices[0].message.content

  def bias_classifier(self, query):
    completion = self.client.chat.completions.create(
    model=self.bias_model_id,
    messages=[
        {"role": "system", "content": """You are a great data classifier conducting on a online comment at news article.
        All you have to do is convert the comments in the query into bias about the comments.
        편향은 그 댓글이 얼마나 편향되어있는지를 나타내며, -1 혹은 1의 값을 가집니다.
        편향성은 흑백논리에 해당되는 "정치", "성별", "세대" 카테고리에만 적용됩니다.
        우파를 비하하는 댓글은 -1, 좌파를 비하하는 댓글은 1입니다.
        역시나 남성을 비하하는 댓글은 -1, 여성을 비하하는 댓글은 1입니다.
        노인 세대를 비하하는 댓글은 -1, 어린이 세대를 비하하는 댓글은 1입니다.
        "정치", "성별", "세대" 카테고리에 해당하지 않는 값은 모두 0으로 표현하십시오.
        따라서 편향 값은 -1부터 1 사이의 실수로 표현됩니다."""},
        {"role": "user", "content": query},
      ]
    )
    return completion.choices[0].message.content

  def aggr_classifier(self, query):
    completion = self.client.chat.completions.create(
    model=self.aggr_model_id,
    messages=[
        {"role": "system", "content": """You are a great data classifier conducting on a online comment at news article.
        All you have to do is convert the comments in the query into aggression about the comments."""},
        {"role": "user", "content": query},
      ]
    )
    mapping = {'none': 0.0, 'offensive': 0.5, 'hate': 1.0}
    try:
      val = mapping[completion.choices[0].message.content]
    except:
      val = np.nan
    return val

#TODO: 뉴스 기사 종합 분석하는 클래스
import pickle
from gensim.models import KeyedVectors

from khaiii import KhaiiiApi

class analyzeModule(langchainModule, ClassifierSet, APIkeyImport):
    def __init__(self):
        langchainModule.__init__(self)
        ClassifierSet.__init__(self)
        self.cluster_model = pickle.load(open("model/clustering/comment_feature_clustered_model.pkl", "rb"))
        self.w2v_model = KeyedVectors.load_word2vec_format('model/w2v/comment_w2v', binary=False, encoding='utf-8')

    def calculate_relation(self, content, comments):
        return langchainModule.calculate_relation(self, content, comments)[1]

    def summarize_content(self, title, description, content):
        return langchainModule.summarize_content(title, description, content)[1]

    def comments_processing(self, comments_list):
        morph_analyze = KhaiiiApi()
        cate_to_int_dict = {'정치': 0, '인종': 1, '성별': 2, '외모': 3, '세대': 4, '기타': 5}
        processed_comments = []

        for comment in comments_list:
            if comment == '' or comment == None:
                continue
            
            temp_lex = []
            for word in morph_analyze.analyze(comment):
                for element in word.morphs:
                    temp_lex.append(element.lex)

            vector = []
            for word in temp_lex:
                try:
                    vector.append(self.w2v_model.get_vector(word)) # word2vec 모델에 없는 단어는 제외, 모델 구성 시 min_count로 제외되었을 수 있기 때문
                except:
                    pass
            
            vector = np.mean(vector)

            cate = ClassifierSet.cate_classifier(self, comment)
            try:
                cate_encoded = cate_to_int_dict[cate]
            except:
                cate_encoded = cate_to_int_dict['기타']
            bias = ClassifierSet.bias_classifier(self, comment)
            try:
                bias = int(bias)
            except:
                bias = 0
            aggr = ClassifierSet.aggr_classifier(self, comment)

            comment_vec = np.array([cate_encoded, bias, aggr, vector]).reshape(1, -1)
            cmt = {"comment": comment, "category": cate, "category_encoded": cate_encoded, "bias": bias, "aggression": aggr, "cluster": self.cluster_model.predict(comment_vec).tolist()[0]}
            processed_comments.append(cmt)

        return processed_comments
    
    def report_article(self, title, description, content, comments_list):
        comments_analytics = self.comments_processing(comments_list)
        return langchainModule.report(self, title, description, content, comments_analytics)[1]