"""Main module."""
# %%
import os
try:

    from LLM.request_llm import get_response
    from base_prompt import base_prompt as BASEPROMPT
    from tweets_topic.get_topic import topic_extraction
    from tweets_similarity_search.search_tweets import TweetsSimilarity
    from tweets_irony.get_irony import irony_extraction
    from tweets_sentiment.get_sentiment import sentiment_extraction
    from tweets_emotion.get_emotions import emotion_extraction

    from tweets_loader.csv_tweets_loader import loadCSVTweets

except ImportError:
    from .base_prompt import base_prompt as BASEPROMPT
    from .LLM.request_llm import get_response
    from .tweets_topic.get_topic import topic_extraction
    from .tweets_similarity_search.search_tweets import TweetsSimilarity
    from .tweets_irony.get_irony import irony_extraction
    from .tweets_sentiment.get_sentiment import sentiment_extraction
    from .tweets_emotion.get_emotions import emotion_extraction

    from .tweets_loader.csv_tweets_loader import loadCSVTweets

# %%
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI, PromptTemplate
from langchain.chains.question_answering import load_qa_chain
# from langchain.embeddings import ModelScopeEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from chromadb.config import Settings
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import json
import requests
import json
import time
# %%
from langchain.embeddings import HuggingFaceEmbeddings

# %%


class TwitterPersona:

    def __init__(self,
                 personality_username: str,
                 persona_name="Elon Musk",
                 debug=False) -> None:

        self.NAME = persona_name
        self.debug = debug
        self.personality_username = personality_username

        self.persona_tweets = self.get_tweets(self.personality_username)[:50]

        self.tweets_similarity = TweetsSimilarity(self.persona_tweets,
                                                  debug=self.debug)
        
        self.get_all_snippets()

    def get_tweets(self, personality_username):

        return loadCSVTweets(cleanTweets=True)

    def get_emotions_prompt_snippet(self):

        _, emotions_percentage_dict = emotion_extraction(self.persona_tweets,
                                                         thresh=0.7,
                                                         top_k=3,
                                                         debug=self.debug)

        emotions_percentages = [[k, v*100]
                                for k, v in emotions_percentage_dict.items()]

        line_seperated_emotions_percentages = "\n".join(
            [f"- {k.replace('_', ' ')}: {(v):.2f}%" for k, v in emotions_percentages])

        emotions_prompt_snippet = """\nEmotions:\n{}\n""".format(line_seperated_emotions_percentages)

        return emotions_prompt_snippet

    def get_topic_prompt_snippet(self):

        _, topic_percentages_dict = topic_extraction(self.persona_tweets,
                                                     thresh=0.7,
                                                     debug=self.debug)

        line_seperated_topic_percentages = "\n".join(
            [f"- {k.replace('_', ' ')}: {(v*100):.2f}%" for k, v in topic_percentages_dict.items()])

        topic_prompt_snippet = """Topic:\n{}\n""".format(line_seperated_topic_percentages)

        return topic_prompt_snippet

    def get_sentiment_prompt_snippet(self):

        _, sentiment_percentages_dict = sentiment_extraction(self.persona_tweets,
                                                             debug=self.debug)

        line_seperated_sentiment_percentages = "\n".join(
            [f"- {k.replace('_', ' ')}: {(v*100):.2f}%" for k, v in sentiment_percentages_dict.items()])

        sentiment_prompt_snippet = """\nSentiments:\n{}\n""".format(line_seperated_sentiment_percentages)

        return sentiment_prompt_snippet

    def get_irony_prompt_snippet(self):

        _, irony_percentages_dict = irony_extraction(self.persona_tweets,
                                                     thresh=0.7,
                                                     debug=self.debug)

        line_seperated_irony_percentages = "\n".join(
            [f"- {k.replace('_', ' ')}: {(v*100):.2f}%" for k, v in irony_percentages_dict.items()])

        irony_prompt_snippet = """\nIrony:\n{}\n""".format(line_seperated_irony_percentages)

        return irony_prompt_snippet
    
    def get_all_snippets(self):
        
        print(f"Analysing {self.NAME}'s {len(self.persona_tweets)} Tweets for Personality Analysis...")
        print("Please Wait. This may take a while...\n")

        self.emotions_prompt_snippet = self.get_emotions_prompt_snippet()
        self.topic_prompt_snippet = self.get_topic_prompt_snippet()
        self.sentiment_prompt_snippet = self.get_sentiment_prompt_snippet()
        self.irony_prompt_snippet = self.get_irony_prompt_snippet()

        self.all_snippets = "\n".join([self.emotions_prompt_snippet,
                                       self.topic_prompt_snippet,
                                       self.sentiment_prompt_snippet,
                                       self.irony_prompt_snippet])
        self.all_snippets = f"{self.NAME}'s Tweet Analysis (Percentages):" + self.all_snippets

    def get_similar_tweets(self, query_tweet):

        matching_tweets = self.tweets_similarity.get_similar_tweets(query_tweet,
                                                                    thresh=0.7,
                                                                    top_k=5)

        
        line_seperated_tweets = "\n".join(matching_tweets)

        similar_tweets_prompt_snippet = """Their Similar Tweets to the user query:\n{}\n""".format(line_seperated_tweets)

        if line_seperated_tweets == "":
            similar_tweets_prompt_snippet = ""
        return similar_tweets_prompt_snippet


# %%


class ConversationBot:

    def __init__(self,
                 persona_object,
                 user_name="Max",
                 debug=False) -> None:

        self.user_name = user_name

        self.debug = debug
        self.persona = persona_object

        self.NAME = self.persona.NAME
        
        self.all_snippets = self.persona.all_snippets

        self.personality_username = self.persona.personality_username

        # self.embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-xl", model_kwargs=model_kwargs,)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cuda:0'})

        HOME = os.path.expanduser('~')

        twitterpersona_path = os.path.join(HOME,
                                           ".twitterpersona")
        # '/.db_instructor_xl'

        self.persist_directory = os.path.join(
            twitterpersona_path,
            f'.{self.user_name}_{self.personality_username}_db')
        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory, exist_ok=True)

        CHROMA_SETTINGS = Settings(
            chroma_db_impl='duckdb+parquet',
            persist_directory=self.persist_directory,
            anonymized_telemetry=False
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        curr_file_path = os.path.dirname(os.path.realpath(__file__))

        
        prev_converstion_path = os.path.join(
            twitterpersona_path, f"{self.user_name}_{self.personality_username}_prev_conversation.txt")
        
        if not os.path.exists(twitterpersona_path):
            os.makedirs(twitterpersona_path, exist_ok=True)

        if not os.path.exists(prev_converstion_path):
            with open(prev_converstion_path, "w") as f:
                f.write(f"{self.user_name}: . . .")

        loader = TextLoader(prev_converstion_path)
        texts_ = loader.load()
        texts = self.text_splitter.create_documents([texts_[0].page_content])
        db = Chroma.from_documents(
            texts, self.embeddings, persist_directory=self.persist_directory)
        db.persist()

        self.answers = []
        self.conv_buffer = []

        self.avg_retrival_time = 0
        self.avg_generation_time = 0
        self.avg_store_time = 0
        self.avg_total_time = 0
        self.texts = []

        self.count = 0
        
        
    def reply_user(self, query):

        self.count += 1

        starting_time = time.time()
        query = str(query)
        # print(f"{self.user_name}: {query}\n")
        self.conv_buffer.append(f"{self.user_name}: {query}")
        vectordb = Chroma(persist_directory=self.persist_directory,
                          embedding_function=self.embeddings)

        # without similarity score
        context = vectordb.similarity_search_with_score(query)
        retriever = vectordb.as_retriever(search_type="mmr")
        context = retriever.get_relevant_documents(query)
        context = context[0].page_content

        # print(f"""Context Fetched for query: "{query}" :\n""",context)
        retrival_time = time.time() - starting_time

        # print(f"Retrieval Time: {retrival_time}")
        time_after_retrieval = time.time()

        last_convo = self.conv_buffer[-20:]
        last_convo_str = "\n".join(last_convo)

        

        matching_tweets = self.persona.get_similar_tweets(query)


        base_prompt = BASEPROMPT.format(self.NAME, 
                                        self.personality_username,
                                        self.user_name)

        system_prompt = f"""
        {base_prompt}
        
        {self.all_snippets}
        
        {matching_tweets}
        
        The memory of the previous conversation between {self.NAME} and {self.user_name}:
        {context}

        Continuing from the last conversation where {self.NAME} greets {self.user_name} with reference to the last conversation:
        {last_convo_str}\n"""

        message = query
        prompt = f"{system_prompt}{self.NAME}:"
        
        while "\n\n" in prompt:
            prompt = prompt.replace("\n\n", "\n")
        
        while "    " in prompt:
            prompt = prompt.replace("    ", "")

        
        
        self.prompt = prompt

        # PROMTING THE LLM HERE
        response = get_response(prompt)

        generation_time = time.time() - time_after_retrieval

        person = response.replace(prompt, "").split("/n")[0]

        if person.rfind(":") > 0:
            person = person[:person.rfind(":")]

        if person[-1] not in ['.', '?', '!']:
            if person.rfind(".") > person.rfind("?") and person.rfind(".") > person.rfind("!"):
                # print(f"Truncated from response: {person[person.rfind('.')+1:]}")
                person = person[:person.rfind(".")] + person[person.rfind(".")]
            elif person.rfind("?") > person.rfind(".") and person.rfind("?") > person.rfind("!"):
                # print(f"Truncated from response: {person[person.rfind('.')+1:]}")
                person = person[:person.rfind("?")] + person[person.rfind("?")]

            elif person.rfind("!") > person.rfind(".") and person.rfind("!") > person.rfind("?"):
                # print(f"Truncated from response: {person[person.rfind('.')+1:]}")
                person = person[:person.rfind("!")] + person[person.rfind("!")]

        len_response = len(person.split())

        time_after_generation = time.time()

        # print(f"""{self.NAME}: {person} \n\n""")
        self.conv_buffer.append(f"{self.NAME}: {person}")
        self.answers.append(person)

        dialog = f"{self.user_name}: {query}\n{self.NAME}: {person}\n"
        self.texts.append(dialog)
        if self.count % 3 == 0:
            texts_str = "\n".join(self.texts)

            textss = self.text_splitter.create_documents([texts_str])
            db = Chroma.from_documents(
                textss, self.embeddings, persist_directory=self.persist_directory)
            db.persist()
            self.texts = []

        store_time = time.time() - time_after_generation

        return person
# %%


if __name__ == "__main__":
    
    elonpersona = TwitterPersona("elonmusk")
    bot = ConversationBot(elonpersona)

    with open("../sample_data/sampleQuestions.txt") as f:
        questions = f.readlines()

    for question in questions:
        A = bot.reply_user(question)
        print(f"{bot.user_name}: {question}Elon Musk:{A} \n\n")
# %%
