import cohere
import pandas as pd
from tqdm import tqdm

from config import COHERE_API_KEY, NEWS_SAMPLE_DATASET, DATA


def main():
    df = pd.read_csv(NEWS_SAMPLE_DATASET)
    cohere_client = cohere.Client(COHERE_API_KEY)

    model = "small"
    batch_size = 96
    batch = []
    vectors = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        batch.append(row["text"])

        if len(batch) >= batch_size:
            response = cohere_client.embed(texts=batch, model=model)
            vectors.append(response.embeddings)
            batch = []

    if len(batch) > 0:
        response = cohere_client.embed(texts=batch, model=model)
        vectors.append(response.embeddings)
        batch = []

    df["vector"] = [item for sublist in vectors for item in sublist]

    df.to_csv(DATA / "news_sample_with_vectors.csv", index=False)


if __name__ == "__main__":
    main()
