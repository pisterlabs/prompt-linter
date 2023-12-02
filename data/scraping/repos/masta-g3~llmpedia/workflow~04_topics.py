import json
import sys, os
import pandas as pd
import re
import openai
import warnings
from dotenv import load_dotenv

load_dotenv()
PROJECT_PATH = os.environ.get("PROJECT_PATH")
sys.path.append(PROJECT_PATH)
os.chdir(PROJECT_PATH)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

from umap import UMAP
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import OpenAI, MaximalMarginalRelevance
from hdbscan import HDBSCAN
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

import utils.paper_utils as pu
import utils.db as db

db_params = pu.db_params

## Download necessary NLTK data.
nltk.download("wordnet")
nltk.download("stopwords")

REFIT = False

## Create a lemmatizer and list of stop words.
LEMMATIZER = WordNetLemmatizer()
STOP_WORDS = list(set(stopwords.words("english")))

PROMPT = """
I have a topic that contains the following Large Language Model related documents:
[DOCUMENTS]
The topic is described by the following keywords: [KEYWORDS]

Based on the information above, extract a short but highly descriptive topic label of at most 5 words. Do not use "Large Language Model" in your description. Make sure it is in the following format:
topic: <topic label>
"""


def process_text(text: str) -> str:
    """Preprocess text."""
    text = text.lower()
    text = re.sub(r"\W", " ", text)
    text = " ".join(
        [LEMMATIZER.lemmatize(word) for word in text.split() if word not in STOP_WORDS]
    )
    return text


def load_and_process_data(title_map: dict) -> pd.DataFrame:
    """Load and process data from json files, return DataFrame."""
    df = pd.DataFrame(columns=["title", "summary", "main_contribution", "takeaways"])
    for arxiv_code, title in title_map.items():
        fpath = os.path.join(PROJECT_PATH, "data", "summaries", f"{arxiv_code}.json")
        with open(fpath) as f:
            summary = json.load(f)
        df.loc[str(arxiv_code)] = [
            title,
            summary["Summary"],
            summary["main_contribution"],
            summary["takeaways"],
        ]
    return df


def create_embeddings(df: pd.DataFrame) -> tuple:
    """Create embeddings."""
    content_cols = ["summary", "main_contribution", "takeaways"]
    df_dict = (
        df[content_cols].apply(lambda x: "\n".join(x.astype(str)), axis=1).to_dict()
    )
    all_content = list(df_dict.values())
    embedding_model = SentenceTransformer("barisaydin/gte-large")
    embeddings = embedding_model.encode(all_content, show_progress_bar=True)
    return all_content, embedding_model, embeddings


def create_topic_model(embeddings: list, prompt: str) -> BERTopic:
    """Create topic model."""
    load_dotenv()
    umap_model = UMAP(
        n_neighbors=10, n_components=8, min_dist=0.0, metric="cosine", random_state=200
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=15,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer_model = CountVectorizer(
        stop_words=STOP_WORDS,
        ngram_range=(2, 3),
        min_df=1,
        max_df=0.8,
        preprocessor=process_text,
    )
    mmr_model = MaximalMarginalRelevance(diversity=0.5)
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai_model = OpenAI(
        # model="gpt-3.5-turbo-1106",
        model="gpt-4-1106-preview",
        exponential_backoff=True,
        chat=True,
        prompt=prompt,
        nr_docs=8,
    )
    topic_model = BERTopic(
        embedding_model=embeddings,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=[mmr_model, openai_model],
        top_n_words=10,
        verbose=True,
    )
    return topic_model


def extract_topics_and_embeddings(
    all_content: list,
    embeddings: list,
    topic_model: BERTopic,
    reduced_model: UMAP,
    refit=False,
) -> tuple:
    """Extract topics and embeddings."""
    if refit:
        topics, _ = topic_model.fit_transform(all_content, embeddings)
        reduced_embeddings = reduced_model.fit_transform(embeddings)
    else:
        topics, _ = topic_model.transform(all_content, embeddings)
        reduced_embeddings = reduced_model.transform(embeddings)

    topic_dists = db.get_topic_embedding_dist()
    reduced_embeddings[:, 0] = (
        reduced_embeddings[:, 0] - topic_dists["dim1"]["mean"]
    ) / topic_dists["dim1"]["std"]
    reduced_embeddings[:, 1] = (
        reduced_embeddings[:, 1] - topic_dists["dim2"]["mean"]
    ) / topic_dists["dim2"]["std"]

    return topics, reduced_embeddings, reduced_model


def store_topics_and_embeddings(
    df: pd.DataFrame,
    topics: list,
    reduced_embeddings: list,
    topic_model: BERTopic,
    reduced_model: UMAP,
    refit=False,
):
    """Store topics and embeddings."""
    if refit:
        topic_model.save(
            "data/topic_model.pkl",
            save_ctfidf=True,
            save_embedding_model=True,
            serialization="pickle",
        )
        pd.to_pickle(reduced_model, "data/reduced_model.pkl")

    topic_names = topic_model.get_topic_info().set_index("Topic")["Name"]
    topic_names[-1] = "Miscellaneous"
    clean_topic_names = [
        topic_names[t].split("_")[-1].replace('"', "").strip() for t in topics
    ]
    df["topic"] = clean_topic_names
    df["dim1"] = reduced_embeddings[:, 0]
    df["dim2"] = reduced_embeddings[:, 1]
    # topic_path = os.path.join(PROJECT_PATH, "data", "topics.pkl")
    # df[["topic", "dim1", "dim2"]].to_pickle(topic_path)
    df.index.name = "arxiv_code"
    df.reset_index(inplace=True)
    if_exists_policy = "replace" if refit else "append"
    db.upload_df_to_db(
        df[["arxiv_code", "topic", "dim1", "dim2"]],
        "topics",
        pu.db_params,
        if_exists=if_exists_policy,
    )


def main():
    """Main function."""
    title_map = db.get_arxiv_title_dict(db_params)
    if REFIT:
        df = load_and_process_data(title_map)
        all_content, embedding_model, embeddings = create_embeddings(df)
        topic_model = create_topic_model(embedding_model, PROMPT)
        reduced_model = UMAP(
            n_neighbors=12,
            n_components=2,
            min_dist=0.0,
            metric="cosine",
            random_state=200,
        )

    else:
        ## Predict topics on new documents using existing model.
        topic_model = BERTopic.load("data/topic_model.pkl")
        reduced_model = pd.read_pickle("data/reduced_model.pkl")

        arxiv_codes = db.get_arxiv_id_list(db_params, "summaries")
        done_codes = db.get_arxiv_id_list(db_params, "topics")
        working_codes = list(set(arxiv_codes) - set(done_codes))
        title_map = {k: v for k, v in title_map.items() if k in working_codes}

    df = load_and_process_data(title_map)
    all_content, embedding_model, embeddings = create_embeddings(df)
    topics, reduced_embeddings, reduced_model = extract_topics_and_embeddings(
        all_content, embeddings, topic_model, reduced_model, refit=REFIT
    )
    store_topics_and_embeddings(
        df, topics, reduced_embeddings, topic_model, reduced_model, refit=REFIT
    )

    print("Done!")


if __name__ == "__main__":
    main()
