import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import Weaviate
from tqdm import tqdm
from weaviate_client import client

tqdm.pandas(desc="Processing embeddings")


def download_imdb_ratings() -> pd.DataFrame:
    url = "https://datasets.imdbws.com/title.ratings.tsv.gz"
    return pd.read_csv(url, sep="\t")


def add_imdb_ratings(movies: pd.DataFrame, imdb_ratings: pd.DataFrame) -> pd.DataFrame:
    # Some movies have no IMDb id (None), so we need m:1
    imdb_formatted = imdb_ratings.rename(
        columns={
            "tconst": "imdb_id",
            "averageRating": "imdb_vote_average",
            "numVotes": "imdb_vote_count",
        }
    )

    merged = movies.merge(
        imdb_formatted,
        on="imdb_id",
        how="left",
        validate="m:1",
    )
    return merged


def read_movies(source: str) -> pd.DataFrame:
    res = pd.read_parquet(source)
    return res.assign(
        providers=lambda df: df["providers"].apply(np.ndarray.tolist),
        genres_list=lambda df: df["genres_list"].str.split(", "),
        release_year=lambda df: pd.to_datetime(df["release_date"]).dt.year,
    )


def parse_null_float(val: float) -> float | None:
    if np.isnan(val):
        return None
    return val


def parse_null_int(val: int) -> int | None:
    if np.isnan(val):
        return None
    return int(val)


def create_documents(data: pd.DataFrame) -> list[Document]:
    docs = []
    for _, row in data.iterrows():
        properties = {
            "show_id": row["id"],
            "title": row["title"],
            "release_year": parse_null_int(row["release_year"]),
            "genres": row["genres_list"],
            "trailer_url": row["trailer"],
            "watch": row["provider_url"],
            "providers": row["providers"],
            "vote_average": parse_null_float(row["vote_average"]),
            "vote_count": row["vote_count"],
            "imdb_vote_average": parse_null_float(row["imdb_vote_average"]),
            "imdb_vote_count": parse_null_int(row["imdb_vote_count"]),
            "runtime": row["runtime"],
        }
        doc = Document(page_content=row["overview"], metadata=properties)
        docs.append(doc)
    return docs


def main():
    load_dotenv()
    DATA_SOURCE = "data/final_movies.parquet"
    movies = read_movies(DATA_SOURCE)
    imdb_ratings = download_imdb_ratings()
    moviews_with_imbd_ratings = add_imdb_ratings(movies, imdb_ratings)

    docs = create_documents(moviews_with_imbd_ratings)

    embeddings = OpenAIEmbeddings()
    Weaviate.from_documents(
        docs,
        embeddings,
        index_name="Movie",
        client=client,
        text_key="overview",
    )


if __name__ == "__main__":
    main()
