import numpy as np
import pandas as pd
import streamlit as st
from streamlit.connections import ExperimentalBaseConnection
import openai
import os
import torch
import tweepy
from typing import List, Tuple, Union
import nltk
from dotenv import load_dotenv
from nltk.sentiment import SentimentIntensityAnalyzer

# Load environment variables from .env file
load_dotenv()

nltk.download('vader_lexicon')

# Get the OpenAI API key from the environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not found.")
openai.api_key = openai_api_key


class TwitterConnection(st.connections.ExperimentalBaseConnection):
    """
    Handles connections to the Twitter API.

    Inherits from Streamlit's ExperimentalBaseConnection. This class overrides the _connect and query methods.

    Attributes:
        secrets (dict): Twitter API keys.
        api (tweepy.API): Tweepy API object for Twitter interactions.

    Methods:
        _connect: Establishes and returns the underlying connection object.
        query: Fetches tweets containing a specified search term.
    """
    
    def __init__(self):
        """
        Fetches Twitter API keys from different sources and stores them as instance attributes.
        Raises an exception if the required API keys are not set.
        """
        try:
            self.secrets = st.secrets["twitter"]
        except KeyError:
            self.secrets = {
                "api_key": os.getenv("TWITTER_API_KEY"),
                "api_key_secret": os.getenv("TWITTER_API_KEY_SECRET"),
                "access_token": os.getenv("TWITTER_ACCESS_TOKEN"),
                "access_token_secret": os.getenv("TWITTER_ACCESS_TOKEN_SECRET"),
            }

        if not all(self.secrets.values()):
            raise Exception("Twitter API keys are not set. Please set them as environment variables or Streamlit secrets.")

        self.api = None

    def _connect(self):
        """
        Establishes a connection to the Twitter API.

        Returns:
            tweepy.API: Tweepy API object for Twitter interactions.
        Raises an exception if there's an issue with the connection.
        """
        try:
            auth = tweepy.OAuthHandler(self.secrets["api_key"], self.secrets["api_key_secret"])
            auth.set_access_token(self.secrets["access_token"], self.secrets["access_token_secret"])
            self.api = tweepy.API(auth)
            st.toast("Connected to 𝕏 API")
            return self.api
        except Exception as e:
            st.error(f"Failed to connect to 𝕏 API: {e}")
            return None

    @staticmethod
    @st.cache_data
    def query(api: tweepy.API, search_term: str, count: int = 10) -> List[str]:
        """
        Fetches the most recent tweets containing the given search term.

        Args:
            api (tweepy.API): Tweepy API object for Twitter interactions.
            search_term (str): Term to search for in tweets.
            count (int, optional): Number of tweets to fetch. Defaults to 10.

        Returns:
            list[str]: List of tweets containing the search term.
        """
        try:
            tweets = tweepy.Cursor(api.search_tweets, q=search_term, lang="en", result_type="recent").items(count)
            return [tweet.text for tweet in tweets]
        except tweepy.RateLimitError as e:
            st.toast("Twitter API rate limit has been reached. Please try again later.", icon="⏳")
            return []
        except Exception as e:
            st.error(f"Failed to fetch tweets: {e}")
            return []

@st.cache_resource
def get_env_variable(name, default=None):
    """
    Retrieves the value of a specified environment variable.

    First, it attempts to fetch the variable from Streamlit secrets. If the variable is not available there, 
    it falls back to the local environment. If the environment variable is not found in either place, 
    it returns the default value.

    Args:
        name (str): The name of the environment variable.
        default (str, optional): The default value to return if the environment variable is not found.

    Returns:
        str: The value of the environment variable or the default value.
    """
    if st.secrets is not None and name in st.secrets:
        # Fetch the secret from Streamlit secrets
        return st.secrets[name]
    else:
        # Try to get the secret from the local environment
        value = os.getenv(name)

        if value is None and default is not None:
            # If the environment variable is not found and a default value is provided,
            # return the default value
            return default
        else:
            return value

st.set_page_config(
    page_title="Sent𝕏Ment: Enlightened Sentiment Analysis",
    page_icon="🔎",
    layout="centered",
    menu_items={
        'Get Help': 'https://github.com/AdieLaine/SentXMent',
        'Report a Bug': 'https://github.com/AdieLaine/SentXMent/issues',
        'About': """
            ## Sent𝕏Ment: Enlightened Sentiment Analysis
            Embark on a journey of emotional understanding with Sent𝕏Ment - a sophisticated sentiment analysis application. Sent𝕏Ment empowers you to delve into the emotional subtext of various forms of written communication, be it tweets, articles, reviews, or personal texts.
            
            ## The Power of Sentiment Analysis
            Sentiment Analysis, often known as opinion mining, uncovers the emotions and sentiments underlying the written word. It serves as a compass, guiding you through the vast ocean of public opinion, customer feedback, and social sentiment.
            
            ## How Sent𝕏Ment Elevates Sentiment Analysis
            Sent𝕏Ment harnesses the cutting-edge advancements in Natural Language Processing (NLP) and Machine Learning to provide comprehensive sentiment analysis. It employs the NLTK SentimentIntensityAnalyzer for initial sentiment detection, generating a sentiment score that represents the text's emotional polarity.

            But Sent𝕏Ment doesn't stop at numbers. It interprets these scores using the state-of-the-art OpenAI GPT-3.5 Turbo model, bringing AI-powered insight to sentiment analysis. This approach not only quantifies sentiment but also provides a qualitative understanding of the text's emotional context.

            Beyond analysis, Sent𝕏Ment offers feedback and suggestions based on the sentiment analysis results. Crafted from a perspective influenced by analytical psychology and logical reasoning, this feedback gives you a deeper understanding of the sentiment conveyed in the text.
            
            ## Expanding Horizons with Sent𝕏Ment
            - Gauge public sentiment on trending topics through Twitter sentiment analysis.
            - Understand customer sentiment by analyzing product reviews.
            - Keep a pulse on public perception by monitoring sentiment in news articles.

            ## Be a Part of Sent𝕏Ment's Journey
            Sent𝕏Ment is a community-driven, open-source project. We invite developers, data scientists, AI enthusiasts, and curious minds to contribute to its growth. Explore our [GitHub](https://github.com/your-github-username/SentXMent) repository to learn more about the project and discover how you can contribute.

            Dive into the world of sentiment analysis with Sent𝕏Ment. Let's explore the emotional depth of the written word together.
        """
    }
)


def apply_css_and_display_title():
    """
    Applies a custom CSS style to the title and displays the title.

    The function creates a custom CSS style for the title, breaking it into three parts: 'Sent', '𝕏', and 'Ment', 
    each with a unique color. It then applies this style to the title and displays the title on the webpage 
    using Streamlit's markdown functionality. Following the title, a 'Sentiment Analysis' subtitle is displayed.
    """
    title_style = """
        <style>
            .title-text {
                text-align: center;
                margin-bottom: 0;
                padding: 10px;
                font-size: 59px;
                font-smoothing: antialiased;
                -webkit-font-smoothing: antialiased;
            }
            .letter-s {
                color: MediumSeaGreen;
            }
            .letter-x {
                color: Gainsboro;
                font-size: 69px;
                font-smoothing: antialiased;
                -webkit-font-smoothing: antialiased;
            }
            .letter-ment {
                color: Crimson;
            }
        </style>
    """
    st.markdown(title_style, unsafe_allow_html=True)
    st.markdown('<h1 class="title-text"><span class="letter-s">Sent</span><span class="letter-x">\U0001D54F</span><span class="letter-ment">Ment</span></h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center;">Sentiment Analysis for 𝕏</h3>', unsafe_allow_html=True)
    st.markdown("---")


def update_mean(
    current_mean: torch.Tensor,
    current_weight_sum: torch.Tensor,
    value: torch.Tensor,
    weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Updates the running mean and weight sum using the Welford method for maintaining numerical stability.

    This function is a part of the Welford's online algorithm to compute the mean in a numerically stable way.
    This method is especially useful when dealing with floating-point numbers in large datasets.
    
    Args:
        current_mean (torch.Tensor): The current accumulated mean.
        current_weight_sum (torch.Tensor): The current accumulated weight sum.
        value (torch.Tensor): The new value to be included in the mean.
        weight (torch.Tensor): The weight of the new value.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The updated mean and accumulated weight sum.
    """
    weight = torch.broadcast_to(weight, value.shape)
    current_weight_sum += torch.sum(weight)
    current_mean += torch.sum((weight / current_weight_sum) * (value - current_mean))
    return current_mean, current_weight_sum


def update_stable_mean(
    mean_and_weight_sum: torch.Tensor, 
    value: torch.Tensor, 
    weight: Union[float, torch.Tensor] = 1.0
) -> None:
    """
    Updates the stable mean using the 'update_mean' function.

    Args:
        mean_and_weight_sum (torch.Tensor): A tensor storing the current mean and weight sum.
        value (torch.Tensor): The new value to be included in the mean.
        weight (Union[float, torch.Tensor], optional): The weight of the new value. Defaults to 1.0.

    Returns:
        None
    """
    mean, weight_sum = mean_and_weight_sum[0], mean_and_weight_sum[1]

    if not isinstance(weight, torch.Tensor):
        weight = torch.as_tensor(weight, dtype=value.dtype, device=value.device)

    mean_and_weight_sum[0], mean_and_weight_sum[1] = update_mean(
        mean, weight_sum, value, torch.as_tensor(weight)
    )


def calculate_stable_mean(values: list) -> float:
    """
    Calculates the stable mean of a list of values using the 'update_stable_mean' function.

    Args:
        values (list): A list of values for which to calculate the stable mean.

    Returns:
        float: The stable mean of the values.
    """
    mean_and_weight_sum = torch.zeros(2)
    for value in values:
        update_stable_mean(mean_and_weight_sum, torch.tensor(value))
    return mean_and_weight_sum[0].item()


def analyze_sentiment(input_text: str, sia: SentimentIntensityAnalyzer = None) -> dict:
    """
    Analyzes the sentiment of the input text using NLTK's SentimentIntensityAnalyzer.

    This function calculates polarity scores ranging from -1 to 1 for the input text, where 1 indicates positive sentiment
    and -1 indicates negative sentiment. The scores are based on a lexicon of words that have been preassigned scores that
    denote the sentiment they carry. The function returns a dictionary that includes the compound score (an aggregate sentiment
    score) as well as individual scores for the positive, negative, and neutral sentiment of the text.

    Args:
        input_text (str): The input text to analyze.
        sia (SentimentIntensityAnalyzer, optional): An optional SentimentIntensityAnalyzer object. If none is provided, a new one is created.

    Returns:
        dict: A dictionary containing the sentiment scores of the input text. The keys are 'neg', 'neu', 'pos', and 'compound',
        and the values are the corresponding scores.

    Raises:
        ValueError: If the input text is not a string or is empty.
    """
    if not isinstance(input_text, str):
        raise ValueError("Input text must be a string.")
    if input_text.strip() == "":
        raise ValueError("Input text must not be empty.")

    # Create a SentimentIntensityAnalyzer object if none is provided
    if sia is None:
        sia = SentimentIntensityAnalyzer()

    # Calculate sentiment scores
    sentiment_scores = sia.polarity_scores(input_text)

    return sentiment_scores


def detailed_feedback(sentiment_scores: dict, prompt: str) -> str:
    """
    Generates feedback or suggestions based on the sentiment analysis result.

    This function takes the sentiment scores and the input text, sends them to the GPT-3 model as a prompt, and asks the model
    to generate feedback or suggestions influenced by the analytical psychology of Carl Jung and the logical mindset of Abraham Maslow,
    without directly mentioning their names.

    Args:
        sentiment_scores (dict): A dictionary of sentiment scores obtained from analyzing the input text. The keys are 'neg', 'neu', 'pos', and 'compound',
        and the values are the corresponding scores.
        prompt (str): The original text that was analyzed to obtain the sentiment scores.

    Returns:
        str: The feedback or suggestions generated by the GPT-3 model, influenced by the perspectives of Carl Jung and Abraham Maslow.
    """
    feedback_prompt = f"Given these sentiment scores '{sentiment_scores}' for the input text '{prompt}', provide feedback or suggestions in a manner that subtly embodies analytical psychology and the logical patterns that can be interpreted."

    feedback_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": feedback_prompt},
            {"role": "assistant", "content": "With a perspective of Carl Jung and Maslow's Hyierachy of Needs, provide feedback basded on these values that embrace complexity of human emotions and the desire for the right response, let's delve into these sentiment scores..."}
        ],
        temperature=0.7,
        max_tokens=1000,
        stop=None,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    feedback = feedback_response['choices'][0]['message']['content']
    return feedback


def sentiment_x_interpreter(sentiment_scores: dict, prompt: str) -> str:
    """
    Interprets sentiment scores using OpenAI's GPT-3 model.

    Constructs a prompt that includes the original text and its sentiment scores, then sends this prompt to the GPT-3 model.
    The model generates a detailed analysis of the sentiment in the text.

    Args:
        sentiment_scores (dict): Sentiment scores obtained from analyzing the input text.
            The keys are 'neg', 'neu', 'pos', and 'compound', and the values are the corresponding scores.
        prompt (str): The original text that was analyzed to obtain the sentiment scores.

    Returns:
        str: The interpreted sentiment analysis result generated by the GPT-3 model.
    """
    sentiment_interpretation_prompt = f"The sentiment scores for the input text '{prompt}' are '{sentiment_scores}'. Could you interpret these scores and provide a detailed analysis of the sentiment in the text?"

    sentiment_interpretation_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": sentiment_interpretation_prompt},
            {"role": "assistant", "content": f"Based on the provided '{sentiment_scores}', I will reiterate the {prompt}. Then break down the {prompt} in an analysis and detail each part:"}
        ],
        temperature=0.7,
        max_tokens=1000,
        stop=None,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    sentiment_analysis_result = sentiment_interpretation_response['choices'][0]['message']['content']
    return sentiment_analysis_result


def display_results(sentiment_score: float, stable_mean_sentiment_score: float, sentiment_analysis_result: str, feedback: str) -> None:
    """
    Displays the sentiment score, stable mean of sentiment scores, sentiment analysis result, and feedback on the webpage.

    This function displays the sentiment score and the stable mean of sentiment scores in two separate columns on the webpage.
    It then displays the sentiment analysis result as a chat message. Additionally, the function displays an info box with
    a brief explanation of the methodology used to calculate the stable mean of sentiment scores. It also includes an expander
    with detailed information about the sentiment scores and the algorithms used. Finally, it displays a toast message indicating
    the completion of the sentiment analysis and interpretation.

    Args:
        sentiment_score (float): The sentiment score to be displayed.
        stable_mean_sentiment_score (float): The stable mean of sentiment scores to be displayed.
        sentiment_analysis_result (str): The sentiment analysis result to be displayed.
        feedback (str): The feedback or suggestions to be displayed in an expander.

    Returns:
        None
    """
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Sent𝕏Ment Sentiment Score", value=sentiment_score, delta=float(sentiment_score))
    with col2:
        st.metric(label="Stable Mean Score (Welford's Method)", value=stable_mean_sentiment_score, delta=float(stable_mean_sentiment_score))

    st.markdown("---")

    with st.chat_message("assistant"):
        st.markdown(sentiment_analysis_result)

    with st.expander("Deeper Analysis and Feedback"):
        st.markdown(feedback)
        st.info("**Intended For Research/Educational Use** - Sent𝕏Ment uses logic to simulate and invoke a dynamic response. In this example, we simulate the framework of analytical psychology and the principles of logical reasoning. The feedback and suggestions offered here echo the insights of a hybrid version of Carl Jung and Abraham Maslow.")

    with st.expander("Sentiment Scores and Algorithms"):
        st.markdown("""
        **Sent𝕏Ment Sentiment Score:**
        Sent𝕏Ment uses NLTK's Sentiment Analysis to evaluate the sentiment of a text input. The score is a compound value that signifies the overall sentiment of the text, normalized to range between -1 (most negative) and +1 (most positive).

        **Stable Mean Score (Welford's Method):**
        Sent𝕏Ment calculates a numerically stable mean using the Welford Algorithm. This method, robust to outliers and efficient for large datasets, yields more reliable results when dealing with large quantities of data or data with large variations. This logic is inspired by the aggregation method from the public Twitter Algorithm repository.

        **Sent𝕏Ment Logic:**
        The Sent𝕏Ment process involves performing sentiment analysis on a text input using NLTK, calculating the compound sentiment score, and then calculating the stable mean of this score using the implemented functions. This information is then interpreted using OpenAI's GPT-3 model, providing a detailed analysis of the sentiment in the text.

        **Welford's Algorithm:**
        Welford's Algorithm calculates a numerically stable mean, which is particularly useful in situations dealing with large amounts of data or data with large variations. The aggregation logic used here is inspired by methods used in industry settings, such as social media sentiment analysis, and has been adapted from the public Twitter Algorithm repository.
        """)
    st.toast("Sentiment analysis and interpretation complete.")


def main():
    """
    Main function to run the Sent𝕏Ment sentiment analysis application.

    This function orchestrates the sentiment analysis workflow, which includes:
    1. Applying CSS and displaying the title.
    2. Establishing a connection to the Twitter API using the custom TwitterConnection class.
    3. Allowing the user to control the connection status (connect/disconnect) using a select box in the sidebar.
    4. Checking if NLTK data is downloaded. If not, it is downloaded.
    5. Allowing the user to input a tweet for sentiment analysis.
    6. Performing sentiment analysis on the input tweet using NLTK's SentimentIntensityAnalyzer.
    7. Calculating the sentiment score and stable mean of sentiment scores using the Welford algorithm.
    8. Displaying the sentiment score, stable mean of sentiment scores, sentiment analysis result, and detailed feedback using the `display_results` function.

    The function informs the user about the progress of the sentiment analysis and the status of the Twitter API connection. It also warns the user if NLTK data is not found and downloads it.

    Returns:
        None
    """
    apply_css_and_display_title()

    # Establish connection
    twitter_conn = TwitterConnection()

    # Using session state to persist connection status
    if 'connection_status' not in st.session_state:
        st.session_state['connection_status'] = 0  # Start in a neutral state

    # Add a select box to the sidebar for connection control
    connection_status = st.sidebar.selectbox(
        "𝕏 API Connection Status",
        options=["🌚 None", "🟢 Connect to 𝕏", "🔴 Disconnect from 𝕏"],
        index=st.session_state['connection_status']
    )

    # Updating connection status
    if connection_status == "🟢 Connect to 𝕏":
        if not twitter_conn.api:
            twitter_conn.api = twitter_conn._connect()
            if twitter_conn.api:
                st.session_state['connection_status'] = 1  # Update the connection status
                st.sidebar.info("Connected to 𝕏 API")
                print("Connected to 𝕏 API")
            else:
                st.sidebar.error("Failed to connect to 𝕏 API")
        else:
            st.sidebar.info("Already connected to 𝕏 API")
    elif connection_status == "🔴 Disconnect from 𝕏":
        print("Disconnecting from 𝕏 API")  # Print before disconnection
        twitter_conn.api = None  # Disconnect from 𝕏 API
        st.session_state['connection_status'] = 0  # Update the connection status
        st.sidebar.info("Disconnected from 𝕏")

    # Checking and downloading NLTK data once per session
    if 'nltk_data_checked' not in st.session_state:
        st.session_state['nltk_data_checked'] = False

    if not st.session_state['nltk_data_checked']:
        try:
            nltk.data.find('sentiment/vader_lexicon')
            st.session_state['nltk_data_checked'] = True
        except LookupError:
            nltk.download('vader_lexicon')
            st.toast("NLTK data was missing but has now been downloaded.")
            st.session_state['nltk_data_checked'] = True

    # Sentiment analysis workflow
    prompt = st.chat_input("Paste a Tweet to analyze the sentiment...")
    if prompt:
        with st.spinner("Generating sentiment analysis result..."):
            st.toast("Sent𝕏Ment is verifying sentiment values...", icon="🔍")
            if st.session_state['connection_status'] == 0:
                st.toast("No active 𝕏 API connection. Using local sentiment data for this analysis...", icon="⚠️")
            else:
                st.toast("Using NLTK sentiment data for this analysis...", icon="📚")
            sentiment_scores = analyze_sentiment(prompt.strip())
            sentiment_score = sentiment_scores['compound']
            stable_mean_sentiment_score = calculate_stable_mean([sentiment_score])
            sentiment_analysis_result = sentiment_x_interpreter(sentiment_scores, prompt)
            feedback = detailed_feedback(sentiment_scores, prompt)

            # Display results
            display_results(sentiment_score, stable_mean_sentiment_score, sentiment_analysis_result, feedback)

if __name__ == "__main__":
    main()