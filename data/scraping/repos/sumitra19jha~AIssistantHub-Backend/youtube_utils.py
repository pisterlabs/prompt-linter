import re
import datetime

import openai
from httpx import HTTPError
from isodate import parse_duration
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from api.models.analysis import Analysis
from api.models.search_analysis_rel import SearchAnalysisRel

from api.models.search_query import SearchQuery
from api.utils.db import add_commit_
from api.assets import constants

from config import Config
from api.utils import logging_wrapper

logger = logging_wrapper.Logger(__name__)
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

class YotubeSEOUtils:
    def generate_youtube_search_text_gpt4(user, business_type, target_audience, industry, location):
        try:
            system_prompt = {
                "role": "system",
                "content": "You are a Youtube Search Query Writer assistant. Your primary work is to write search queries based on user input that provides relevant results which will be helpful for users.\n\nYour response should be pointwise."
            }

            user_prompt = {
                "role": "user",
                "content": f"User Input\n```\nBusiness type: {business_type}\nTarget audience: {target_audience}\nIndustry: {industry}\nLocation: {location}\n```"
            }

            assistant_response = openai.ChatCompletion.create(
                model=Config.OPENAI_MODEL_GPT4,
                messages=[system_prompt, user_prompt],
                temperature=0.7,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                user=str(user.id),
            )

            # Extract and format search queries as an array
            search_queries = assistant_response["choices"][0]["message"]["content"].strip().split("\n")
            return search_queries
        except Exception as e:
            logger.exception(str(e))
            return None

    # Uses Youtube V3 API for search based on user Input
    # The current form of Query for Search is:
    #   "{business_type} {target_audience} {industry} {goals}"
    def youtube_search(app, query_arr, seo_id, max_results=5):
        with app.app_context():
            youtube = build('youtube', 'v3', developerKey=Config.GOOGLE_SEARCH_API_KEY)

            video_response = []
            video_ids = []

            # Calculate date one year ago from today
            one_year_ago = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime('%Y-%m-%dT%H:%M:%SZ')

            for query in query_arr:
                try:
                    search_model = SearchQuery.query.filter(
                        SearchQuery.search_query == query,
                        SearchQuery.seo_project_id == seo_id,
                        SearchQuery.type == constants.ProjectTypeCons.enum_youtube,
                    ).first()

                    if search_model is None:
                        search_model = SearchQuery(
                            search_query=query,
                            seo_project_id=seo_id,
                            type=constants.ProjectTypeCons.enum_youtube,
                        )
                        add_commit_(search_model)

                    search_response = youtube.search().list(
                        q=query,
                        part="id,snippet",
                        maxResults=max_results,
                        type='video',
                        videoDefinition='high',
                        order='viewCount',
                        publishedAfter=one_year_ago
                    ).execute()

                    for search_result in search_response.get("items", []):
                        if search_result["id"]["kind"] == "youtube#video":
                            video_id = search_result["id"]["videoId"]
                            if video_id in video_ids:
                                continue

                            video_info = youtube.videos().list(
                                part="snippet,statistics,contentDetails",
                                id=video_id
                            ).execute()["items"][0]

                            video_url = f"https://www.youtube.com/watch?v={video_id}"
                            thumbnail_url = search_result["snippet"]["thumbnails"]["default"]["url"]
                            video_duration = parse_duration(video_info["contentDetails"]["duration"]).total_seconds()

                            video_model = Analysis.query.filter(Analysis.video_id == video_id).first()

                            if video_model is None:
                                video_model = Analysis(
                                    type=constants.ProjectTypeCons.enum_youtube,
                                    video_id=video_id,
                                    video_url=video_url,
                                    thumbnail_url=thumbnail_url,
                                    video_duration=video_duration,
                                    title=search_result["snippet"]["title"],
                                    description=video_info["snippet"]["description"],
                                    channel_title=search_result["snippet"]["channelTitle"],
                                    publish_date=video_info["snippet"]["publishedAt"],
                                    views=int(video_info["statistics"].get("viewCount", "-1")),
                                    likes_count=int(video_info["statistics"].get("likeCount", "-1")),
                                    comments_count=int(video_info["statistics"].get("commentCount", "-1"))
                                )
                                add_commit_(video_model)

                            # Check if the video already exists in the video_response list
                            if video_model not in video_response:
                                video_response.append(video_model)

                            search_video_rel_model = SearchAnalysisRel.query.filter(
                                SearchAnalysisRel.search_query_id == search_model.id,
                                SearchAnalysisRel.analysis_id == video_model.id
                            ).first()

                            if search_video_rel_model is None:
                                search_video_rel_model = SearchAnalysisRel(
                                    search_query_id=search_model.id,
                                    analysis_id=video_model.id,
                                )
                                add_commit_(search_video_rel_model)

                            video_ids.append(video_id)

                except HTTPError as e:
                    logger.exception(str(e))
                    return []

            return video_response

    def preprocess_text(text):
        doc = nlp(text)
        filtered_tokens = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop]
        return filtered_tokens

    def yotube_video_keywords_extraction(video_data):
        title_documents = []
        description_documents = []

        for video in video_data:
            title_tokens = YotubeSEOUtils.preprocess_text(video.title)
            description_tokens = YotubeSEOUtils.preprocess_text(video.description)
            title_documents.append(title_tokens)
            description_documents.append(description_tokens)

        return title_documents, description_documents

    def compute_tfidf_matrix(documents):
        dictionary = Dictionary(documents)
        corpus = [dictionary.doc2bow(doc) for doc in documents]
        tfidf_model = TfidfModel(corpus)
        return tfidf_model, corpus, dictionary

    def identify_top_keywords(tfidf_model, corpus, dictionary, top_n=10):
        top_keywords = []

        for doc in corpus:
            doc_tfidf = tfidf_model[doc]
            sorted_tfidf = sorted(doc_tfidf, key=lambda x: x[1], reverse=True)
            top_terms = [(dictionary[term_id], tfidf_value) for term_id, tfidf_value in sorted_tfidf[:top_n]]
            top_keywords.append(top_terms)

        return top_keywords

    def calculate_video_score(video):
        score = 0

        # Assign weights for each factor
        weight_views = 2
        weight_likes = 1.5
        weight_comments = 1
        weight_duration = 0.5

        # Normalize the factors
        normalized_views = video.views / (video.views + 2)
        normalized_likes = video.likes_count / (video.likes_count + 2)
        normalized_comments = video.comments_count / (video.comments_count + 2)
        normalized_duration = video.video_duration / (video.video_duration + 2)

        # Calculate the score
        score = (
            weight_views * normalized_views +
            weight_likes * normalized_likes +
            weight_comments * normalized_comments +
            weight_duration * normalized_duration
        )

        return score

    def calculate_keyword_score(top_keywords, video_data):
        keyword_scores = {}

        for i, video_keywords in enumerate(top_keywords):
            video = video_data[i]
            video_score = YotubeSEOUtils.calculate_video_score(video)

            for keyword, tfidf_value in video_keywords:
                if keyword not in keyword_scores:
                    keyword_scores[keyword] = 0
                
                keyword_scores[keyword] += tfidf_value * video_score

        return keyword_scores

    def rank_keywords(keyword_scores):
        ranked_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked_keywords

    def is_relevant_keyword(keyword):
        # Remove keywords containing only numbers, ".", "http", or "https"
        if re.match(r"^\d+$|^\.+$|^http$|^https$", keyword):
            return False

        # Remove keywords containing URLs
        if re.search(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", keyword):
            return False

        # TODO: Add more filtering conditions if needed

        return True

    def filter_keywords(ranked_keywords):
        filtered_keywords = [kw for kw, score in ranked_keywords if YotubeSEOUtils.is_relevant_keyword(kw)]
        return filtered_keywords

    def generate_title_templates_gpt4(user, keywords_str, num_templates=5):
        try:
            system_prompt = {
                "role": "system",
                "content": "You are an AI assistant trained to generate relevant and engaging article titles based on a set of keywords. Generate 10 title using the following keywords."
            }

            user_prompt = {
                "role": "user",
                "content": f"Keywords: {keywords_str}"
            }

            assistant_response = openai.ChatCompletion.create(
                model=Config.OPENAI_MODEL_GPT4,
                messages=[system_prompt, user_prompt],
                temperature=0.7,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                user=str(user.id),
            )

            # Extract and format search queries as an array
            search_queries = assistant_response["choices"][0]["message"]["content"].strip().split("\n")
            return search_queries
        except Exception as e:
            return None