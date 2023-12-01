from langchain.document_loaders import UnstructuredWordDocumentLoader
import re
import pandas as pd
import openai
from time import sleep
import os


def fetch_source(topic: str, title: str):
    return "https://forum.nos.pt/"


def create_page(title, content, topic):
    return {
        "id": title.split("]")[0].replace("[", ""),
        "title": title.split("]")[1],
        "content": content,
        "type": "Serviços",
        "all": title.split("]")[1] + " Answer: " + content,
        "source": fetch_source(topic, title),
        "topic": topic,
    }


def process_word_faqs(file: str):
    loader = UnstructuredWordDocumentLoader(file, mode="elements")
    docs = loader.load()
    topic = docs[0].page_content.split(": ")[1]

    pages = []
    title = ""
    content = ""
    for doc in docs[1:]:
        if doc.metadata["category"] == "Title" and doc.page_content.find("?") > -1:
            if len(title) > 0 and len(content) > 0:
                pages.append(create_page(title, content, topic))
            title = doc.page_content.replace("\xa0", " ")
            content = ""
        else:
            content += doc.page_content.replace("\xa0", " ")
    pages.append(create_page(title, content, topic))
    return pages


def process_docs(files: list) -> pd.DataFrame:
    docs = []
    for file in files:
        pages = process_word_faqs(file)
        docs += pages
    return pd.DataFrame(docs)


def load_process_md_file(file_path: str):
    """
    Process a markdown file and return a dataframe with the content
    file_path: str
        path to the markdown file

    return: pd.DataFrame
    """
    df = md_to_df(file_path)
    df = process_base_df(df)

    return df


def process_base_df(df):
    """
    Process a dataframe with markdown content
    df : pd.DataFrame
        dataframe with markdown content

    return: None
    """
    df = df.copy()
    df["type"] = "Serviços"
    df["all"] = df["title"] + " Answer: " + df["content"]
    df["topic"] = (
        df["id"]
        .apply(lambda x: re.search("[a-zA-Z]+", x).group())
        .map(
            {
                "FORUM": "Fórum NOS",
            }
        )
    )
    df["source"] = df.apply(lambda x: fetch_source(x["topic"], x["title"]), axis=1)
    return df


def md_to_df(file_path):
    """
    Process a markdown file and return a dataframe with the content
    file_path: str

    return: pd.DataFrame
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    content += "\n\n## \[END])"
    pattern = r"## \[(.*?)\](.*?)\n\n(.*?)(?=\n\n## \[|\n\n## \[END])"
    # find matches
    matches = re.findall(pattern, content, re.DOTALL)
    df = pd.DataFrame(matches, columns=["id", "title", "content"])
    return df


def get_embedding(message):
    openai.api_type = os.environ.get("OPENAI_API_TYPE")
    openai.api_base = os.environ.get("AZURE_OPENAI_SERVICE")
    openai.api_version = os.environ.get("OPENAI_API_VERSION")
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    return openai.Embedding.create(
        deployment_id=os.environ.get("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
        input=message
    )


def build_embedings_w_topics(
    FAQS: pd.DataFrame, topics_dict: dict, openai: openai
) -> pd.DataFrame:
    FAQS["title_embs"] = (
        FAQS["title"]
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
        .str.lower()
    )

    FAQS["embeddings"] = FAQS["title_embs"].apply(
        lambda x: get_embedding(x)["data"][0]["embedding"]
    )

    FAQS["all_embs"] = (
        FAQS["all"]
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
        .str.lower()
    )

    # TODO: Remove this
    # Right now we cannot perform more than 180 request per minute
    sleep(60)
    FAQS["embeddings_qa"] = FAQS["all_embs"].apply(
        lambda x: get_embedding(x)["data"][0]["embedding"]
    )

    topic_rev = {topics_dict[t]["doc_topic_name"]: t for t in topics_dict}
    FAQS["topic_number"] = FAQS["topic"].map(topic_rev)
    return FAQS
