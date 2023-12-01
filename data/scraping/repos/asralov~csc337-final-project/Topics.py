import json
import os
import openai
import re
from newspaper import Article
from newspaper import Config
from nltk.sentiment import SentimentIntensityAnalyzer
from newsapi import NewsApiClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



# Initalize API keys and other variables
news_api_key = os.environ.get("NEWSAPI_KEY")
openai_api_key = os.environ.get("OPENAI_API_KEY")
SID = SentimentIntensityAnalyzer()


class SearchTopic:
    def __init__(self, topic,similarity_threshold):
        # save topic term and get articles
        self._term = topic
        self._articles = self.get_articles()
        self._similarity_threshold=similarity_threshold

    def get_articles(self, num_loop=1):
        """Function to download articles sorted by time from the newsAPI on a given topic"""
        articles = []
        titles = []
        newsapi = NewsApiClient(api_key=news_api_key)
        # loop set so we can add more articles if needed
        for i in range(num_loop):
            api_response = newsapi.get_everything(
                q=self._term,
                language="en",
                sort_by="publishedAt",
                page_size=100,  # max is 100 allowed by API
            )

            for article in api_response["articles"]:
                # Checks for duplicate articles and if they have a summary
                if article["title"] not in titles and article["description"]:
                    articles.append(
                        {
                            "title": article["title"],
                            "url": article["url"],
                            "description": article["description"],
                            "sentiment": 0,
                            "text": "",
                        }
                    )
                    titles.append(article["title"])

        # Download article text and calculate sentiment
        for article_dict in articles:
            config = Config()
            config.browser_user_agent = "Mozilla/5.0..."
            article = Article(article_dict["url"], config=config)
            try:
                article.download()
                article.parse()
                article_dict["text"] = article.text
                sentiment_scores = SID.polarity_scores(article.text)
                article_dict["sentiment"] = sentiment_scores["compound"]
            except Exception as e:
                print(f"Error downloading article: {e}")

        return articles

    def preprocess_text(self, text):
        """Function to preprocess text for TF-IDF"""
        text = text.lower()
        pattern = r"[^\w\s]"
        preprocessed_text = re.sub(pattern, "", text)
        return preprocessed_text

    def calculate_similarity(self):
        """Convert articles to TF-IDF"""
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(
            [self.preprocess_text(article["text"]) for article in self._articles]
        )
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        return cosine_sim

    def find_article_groups(self):
        """Function to find groups of similar articles"""
        similarity_scores = self.calculate_similarity()
        num_articles = len(similarity_scores)
        article_groups = {}
        group_id = 0
        grouped_articles = set()

        for i in range(num_articles):
            # Skip article if it's already grouped
            if i in grouped_articles:
                continue

            # Store articles similar to the current one
            similar_articles = [i]

            # Compare the current article with every other article
            for j in range(num_articles):
                # If the similarity score exceeds the threshold and is not the same article, add it to the similar articles list
                if i != j and similarity_scores[i][j] > self._similarity_threshold:
                    similar_articles.append(j)
                    grouped_articles.add(j)

            if len(similar_articles) > 1:
                article_groups[group_id] = similar_articles
                group_id += 1

        return article_groups

    def create_prompt(self, articles):
        """Function to create prompt for GPT-3.5"""
        prompt = (
            f'Summarize articles related to {self._term} in a concise format suitable for a high-level briefing. '
            'Return the data in a JSON structure with the following fields: '
            '"title" for a general title, "background" for an overview of the topic, '
            '"summary" for a combined summary of all relevant articles, and "topics" to list applicable categories. '
            'The categories include: economics, technology, politics, health, business, sports, entertainment, science, world. '
            'The summary should encapsulate the key points from all pertinent articles. '
            'Exclude any articles that do not align with the central theme of the topic. '
            'The final output should be concise, providing a succinct debriefing. '
            'The JSON structure should be as follows: {title: String, background: String, summary: String, topics: Array}. '
            'This format is intended for integration into a program. So please provide just the JSON structure, additionally, make sure the topics'
            'are just the ones I provided.')
        for article in articles:
            json_article = self._articles[article]
            prompt += f"\n\nTitle: {json_article['title']}\n"
            prompt += f"Text: {json_article['text']}\n"

        return prompt

    def export_GPT_summaries(self):
        """Function to export GPT-3 summaries to text files, along with the summaries from the articles"""
        jsons=[]
        article_groups = self.find_article_groups()
        for group_id, article_indices in article_groups.items():
            try:
                prompt = self.create_prompt(article_indices)
                summary = self.generate_summary(prompt)
                file_name = f"Articles/group_{group_id}_{self._term}.json"
                urls = [self._articles[article_index]['url'] for article_index in article_indices]
                summary_json = {
                    "GPT_response": json.loads(summary),
                    "sentiment": self._articles[article_indices[0]]["sentiment"],
                    "urls": urls
                }
                with open(file_name, "w", encoding="utf-8") as file:
                    # Write the JSON object to the file
                    json.dump(summary_json, file, indent=4, ensure_ascii=False)
                jsons.append(summary_json)
            except Exception as e:
                print(f"Error exporting topic group: {e}")
        return jsons


    def generate_summary(self, prompt):
        """Function to generate summaries using GPT-3.5-turbo-16k"""
        openai.api_key = openai_api_key
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            if response.get("choices"):
                return (
                    response["choices"][0]
                    .get("message", {"content": ""})["content"]
                    .strip()
                )
            else:
                print("No content in response choices.")
                return ""

        except Exception as e:
            print(f"Error generating summary: {e}")
            return ""
