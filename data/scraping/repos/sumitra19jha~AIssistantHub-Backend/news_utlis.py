import re
import json
import requests
from flask import current_app
import concurrent.futures

import nltk
import openai

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import TfidfModel, LdaModel
from gensim.corpora import Dictionary
from gensim.matutils import corpus2dense
from collections import Counter
from sklearn.cluster import KMeans
from gensim.models.coherencemodel import CoherenceModel
from concurrent.futures import ThreadPoolExecutor

from api.models.analysis import Analysis
from api.models.search_analysis_rel import SearchAnalysisRel
from api.models.search_query import SearchQuery

from api.assets import constants
from api.utils import logging_wrapper
from api.utils.db import add_commit_
from config import Config

logger = logging_wrapper.Logger(__name__)

class AssistantHubNewsAlgo:
    def get_data(project_id):
        searches = (
            SearchQuery.query.filter(
                SearchQuery.seo_project_id == project_id,
                SearchQuery.type == constants.ProjectTypeCons.enum_news,
            ).all()
        )

        searches_ids = [search.id for search in searches]

        analysis_data = (
            SearchAnalysisRel.query.filter(
                SearchAnalysisRel.search_query_id.in_(searches_ids),
            )
        )

        analysis_ids = [analysis.analysis_id for analysis in analysis_data]
        print(analysis_ids)

        analysis_data = (
            Analysis.query.filter(
                Analysis.id.in_(analysis_ids),
                Analysis.type == constants.ProjectTypeCons.enum_news,
            ).all()
        )

        news_data = []

        for analysis in analysis_data:
            response = {
                "id": analysis.id,
                "title": analysis.title,
                "htmlTitle": analysis.html_title,
                "displayLink": analysis.display_link,
                "formattedUrl": analysis.formatted_url,
                "htmlSnippet": analysis.snippet,
                "kind":analysis.kind,
                "link":analysis.link,
                "pagemap":analysis.pagemap,
            }

            news_data.append(response)

        return news_data

    def clean_title(title):
        # Remove characters like "\", "/", and quotes
        cleaned_title = re.sub(r'[\\\/"]', '', title)

        # Remove extra spaces
        cleaned_title = re.sub(r'\s+', ' ', cleaned_title).strip()

        return cleaned_title

    # Preprocessing function
    def preprocess(text):
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        text = text.lower()  # Convert text to lowercase
        tokens = nltk.word_tokenize(text)  # Tokenize words
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stopwords.words('english')]
        return tokens

    # Build a dictionary and a corpus for the articles
    def keywords_titles_builder(news_data):
        # Preprocess the articles
        articles = [article["snippet"] for article in news_data]
        preprocessed_articles = [AssistantHubNewsAlgo.preprocess(article) for article in articles]

        # Build a dictionary and a corpus for the articles
        dictionary = Dictionary(preprocessed_articles)
        corpus = [dictionary.doc2bow(article) for article in preprocessed_articles]

        # Calculate TF-IDF scores
        tfidf_model = TfidfModel(corpus)
        tfidf_corpus = tfidf_model[corpus]

        # Convert the tfidf_corpus to a dense matrix
        dense_tfidf_corpus = corpus2dense(tfidf_corpus, num_terms=len(dictionary)).T


        # Extract top keywords and key phrases
        top_keywords = Counter()
        for doc in tfidf_corpus:
            top_keywords.update({dictionary[word_id]: score for word_id, score in doc})

        # Topic modeling using LDA
        lda_model = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=20)
        
        # Compute the topic coherence score
        coherence_model_lda = CoherenceModel(model=lda_model, texts=preprocessed_articles, dictionary=dictionary, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()

        # Filter out irrelevant topics based on a minimum threshold (e.g., 0.3)
        min_coherence_threshold = 0.3
        if coherence_lda >= min_coherence_threshold:
            topics = lda_model.show_topics(num_topics=5, num_words=5, formatted=False)
        else:
            # Handle the case where the topics are not coherent enough
            print("The topics are not coherent enough. Please try again later.")

        # Clustering using KMeans
        n_clusters = min(5, len(dense_tfidf_corpus))
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        clusters = kmeans.fit_predict(dense_tfidf_corpus)

        # Analyzing trends and events
        trending_topics = []
        for cluster in range(5):
            cluster_articles = [article for article, c in zip(articles, clusters) if c == cluster]
            cluster_keywords = Counter()
            for article in cluster_articles:
                article_keywords = AssistantHubNewsAlgo.preprocess(article)
                cluster_keywords.update(article_keywords)
            trending_topics.append(cluster_keywords.most_common(5))

        return trending_topics

    # Generating content titles
    def generate_title(keywords, user_id, app):
        with app.app_context():
            # Join the top keywords with a comma
            keywords_str = ", ".join([keyword[0] for keyword in keywords])

            system_prompt = {
                "role": "system",
                "content": "You are an AI assistant trained to generate relevant and engaging article titles based on a set of keywords. Generate a title using the following keywords."
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
                user=str(user_id),
            )

            # Extract and format the title
            title = assistant_response["choices"][0]["message"]["content"].strip()
            total_tokens = assistant_response['usage']['total_tokens']
            return title, total_tokens

    # Fetch News search text
    def generate_news_search_text_gpt4(user, business_type, target_audience, industry, location):
        try:
            system_prompt = {
                "role": "system",
                "content": "You are a News Search Query Writer AI assistant. Your primary work is to write search queries on google based on user input that provides relevant news result.\n\nYour response should be a single search query."
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

    # Uses Google News API for search based on user Input
    # The current form of Query for Search is:
    #   "{business_type} {target_audience} {industry} {goals}"
    def fetch_google_news(query_arr, project_id, country_code, num_results=5):
        total_points = 0.0
        news_articles = []
        base_url = 'https://www.googleapis.com/customsearch/v1'

        search_models = SearchQuery.query.filter(
            SearchQuery.seo_project_id == project_id,
            SearchQuery.type == constants.ProjectTypeCons.enum_news,
            SearchQuery.search_query.in_(query_arr)
        ).all()

        existing_search_models = {model.search_query: model for model in search_models}
        search_models_to_add = [query for query in query_arr if query not in existing_search_models]

        for query in search_models_to_add:
            search_model = SearchQuery(
                search_query=query,
                seo_project_id=project_id,
                type=constants.ProjectTypeCons.enum_news,
            )
            add_commit_(search_model)
            existing_search_models[query] = search_model
        
        def fetch_news_for_query(query, app):
            with app.app_context():
                search_model = existing_search_models[query]
                params = {
                    'q': f"{query} AND when:7d",  # Fetch articles from the past 7 days
                    'cx': Config.CUSTOM_SEARCH_ENGINE_ID,
                    'key': Config.GOOGLE_SEARCH_API_KEY,
                    'num': num_results,
                    'sort': 'date',  # Sort results by recency
                    'lr': 'lang_en',  # Fetch articles in English
                    'tbm': 'nws',  # Filter results to news articles only
                    'gl': country_code,
                }

                response = requests.get(base_url, params=params)
                fetched_news_articles = []

                if response.status_code == 200:
                    results = response.json()
                    news_article_items = results.get('items', [])

                    existing_news_articles = Analysis.query.filter(
                        Analysis.link.in_([news_article["link"] for news_article in news_article_items]),
                        Analysis.type == constants.ProjectTypeCons.enum_news,
                    ).all()

                    existing_news_article_links = {article.link: article for article in existing_news_articles}
                    new_news_articles = [
                        item for item in news_article_items if item["link"] not in existing_news_article_links
                    ]

                    for news_article_map in new_news_articles:
                        if news_article_map["title"] == "Untitled":
                            continue
                        
                        model_article = Analysis(
                            type=constants.ProjectTypeCons.enum_news,
                            title=news_article_map["title"],
                            html_title=news_article_map["htmlTitle"],
                            display_link=news_article_map["displayLink"],
                            formatted_url=news_article_map["formattedUrl"],
                            snippet=news_article_map["htmlSnippet"],
                            kind=news_article_map["kind"],
                            link=news_article_map["link"],
                            pagemap=news_article_map["pagemap"],
                        )

                        add_commit_(model_article)
                        fetched_news_articles.append(news_article_map)
                        existing_news_article_links[news_article_map["link"]] = model_article

                    for news_article_map in news_article_items:
                        model_article = existing_news_article_links[news_article_map["link"]]

                        search_news_rel_model = SearchAnalysisRel.query.filter(
                            SearchAnalysisRel.search_query_id == search_model.id,
                            SearchAnalysisRel.analysis_id == model_article.id
                        ).first()

                        if search_news_rel_model is None:
                            search_news_rel_model = SearchAnalysisRel(
                                search_query_id=search_model.id,
                                analysis_id=model_article.id
                            )
                            add_commit_(search_news_rel_model)
                else:
                    logger.exception(f"Error: {response.status_code}")
                    print(f"Error: {response.status_code}")
                    return None

                return fetched_news_articles

        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(fetch_news_for_query, query, current_app._get_current_object()): query for query in query_arr}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    news_articles.extend(result)
        
        total_points = total_points + (len(query_arr) * 0.0005)
        
        return news_articles, total_points