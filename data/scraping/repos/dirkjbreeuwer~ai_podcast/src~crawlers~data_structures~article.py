"""
This module defines the data structure for representing an article scraped from the web.
"""

from typing import List, Dict, Optional
from enum import IntEnum
import os
import logging

# pylint: disable=import-error
from dotenv import load_dotenv
import openai
from magentic import prompt

from src.crawlers.data_structures.content import Content

# Load environment variables from .env file
load_dotenv()

# Check if the API key is loaded
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    logging.warning("OPENAI_API_KEY is not set.")
else:
    openai.api_key = api_key
    logging.info("OPENAI_API_KEY is loaded.")

# Set the environment variable
os.environ["MAGENTIC_OPENAI_MODEL"] = "gpt-3.5-turbo"


class ArticleType(IntEnum):
    """
    Represents the type of an article.
    """

    FOUNDATION_MODEL = 1
    PRODUCT_RELEASE = 2
    FUNDING_ROUND = 3
    OTHER = 4


# pylint: disable=line-too-long
@prompt(
    """Classify the article title: {title}
1. Foundation Model: Releases of new foundation ML models. e.g., "OpenAI Launches GPT-5", "Google releases BERT", "Facebook releases RoBERTa"
2. Product Release: New AI-enhanced products. e.g., "Adobe Introduces Photoshop 15 with AI"
3. Funding Round: AI companies' investments or acquisitions. e.g., "AI Startup DeepTech Secures $50M in Series B"
4. Other: General AI topics, guides. e.g., "Ethical Implications of AI", "How to build a chatbot"
"""
)
# pylint: disable=unused-argument
# pylint: disable=missing-function-docstring
def add_article_type(title: str) -> ArticleType:
    pass  # No function body as this is never executed


@prompt(
    """Summarize the following article into 5 short bullet points: {article_text}
        ---
        Remember to keep one short bullet point per line and start your bullet points with -
        """
)
# pylint: disable=unused-argument
def summarize_article(article_text: str) -> str:
    """
    Summarizes an Article into 5 bullet points

    Args:
        article_text (Article.text): The text of the Article

    Returns:
        str: The summary of the Article
    """


@prompt(
    """Predict the relevance of the following article: {article_title}
        ---
        Relevance is a scaled number between 0 and 100, where 0 marginally relevant and 100 is extremely relevant.
        Our audience is entrepreneurs and investors in the AI space
        They care about important companies like Google, Facebook, and OpenAI
        They care about important people like Elon Musk, Sam Altman, and Andrew Ng
        They care about large funding rounds and acquisitions (>$100M) more than small funding rounds and acquisitions (<$100M)
        They care about strategy more than policy (including things like privacy, ethics, and regulation)
        """
)
# pylint: disable=unused-argument
def predict_article_relevance(article_title: str) -> int:
    """
    Predicts the quality of an article

    Args:
        article_title (Article.tile): The title of the Article

    Returns:
        str: The relevance of the Article ranging from 0 to 100
    """


# pylint: disable=R0902
class Article(Content):  # Here we inherit from Content(ABC)
    """
    Represents an article scraped from the web.

    Attributes:
        url (Optional[str]): The URL of the article.
        loaded_domain (Optional[str]): The domain from which the article was loaded.
        title (str): The title of the article.
        date (str): The publication date of the article.
        author (Optional[List[str]]): The authors of the article.
        description (Optional[str]): A brief description or summary of the article.
        keywords (Optional[str]): Keywords associated with the article.
        lang (Optional[str]): The language of the article.
        tags (Optional[List[str]]): Tags associated with the article.
        image (Optional[str]): The main image URL of the article.
        text (str): The main text content of the article.
        id (str): A unique identifier for the article.
        is_vectorized (bool): Indicates if the article has been vectorized.
        article_type (ArticleType): The type of the article.
        article_relevance (int): The relevance of the article.
        summary (str): The summary of the article.
    """

    # Disabling the too many arguments warning because we want Article
    # to be a comprehensive data model
    # pylint: disable=R0913
    def __init__(
        self,
        title: str,
        text: str,
        date: str,
        _id: Optional[str] = None,
        url: Optional[str] = None,
        loaded_domain: Optional[str] = None,
        author: Optional[List[str]] = None,
        description: Optional[str] = None,
        keywords: Optional[str] = None,
        lang: Optional[str] = None,
        tags: Optional[List[str]] = None,
        image: Optional[str] = None,
        videos: Optional[List[Dict[str, str]]] = None,
        is_vectorized: bool = False,
        article_type: Optional[ArticleType] = None,
        article_relevance: Optional[int] = None,
        summary: Optional[str] = None,
    ):
        """Initializes an Article instance with provided attributes."""
        # Here we need to call the __init__ method of the parent class
        super().__init__(
            title,
            text,
            date,
            _id,
            url,
            loaded_domain,
            author,
            description,
            keywords,
            lang,
            tags,
            image,
        )
        self.videos = videos
        self.is_vectorized = is_vectorized
        self._article_type = article_type
        self._article_relevance = article_relevance
        self._summary = summary

        logging.info("Article instance created for title: %s", self.title)

    def __repr__(self):
        """Returns a string representation of the Article instance."""
        return f"<Article(title={self.title}, url={self.url}, date={self.date})>"

    @property
    def summary(self) -> str:
        """Returns the stored summary without regenerating it."""
        return self._summary

    def get_summary(self) -> str:
        """Returns a summary of the article"""
        logging.info("Generating summary for article: %s", self.title)
        self._summary = summarize_article(self.text)
        return self._summary

    @property
    def article_type(self) -> ArticleType:
        """Returns the stored article type without regenerating it."""
        return self._article_type

    def get_type(self) -> ArticleType:
        """Returns the type of the article"""
        logging.info("Classifying article type for: %s", self.title)
        self._article_type = add_article_type(self.title)
        return self._article_type

    @property
    def article_relevance(self) -> int:
        """Returns the stored article relevance without regenerating it."""
        return self._article_relevance

    def get_relevance(self) -> int:
        """Returns the relevance of the article"""
        logging.info("Predicting relevance for article: %s", self.title)
        self._article_relevance = predict_article_relevance(self.title)
        return self._article_relevance

    @property
    def article_id(self):
        """Property to get the article's ID."""
        return self._id

    @article_id.setter
    def article_id(self, value):
        """Setter for the article's ID."""
        self._id = value
