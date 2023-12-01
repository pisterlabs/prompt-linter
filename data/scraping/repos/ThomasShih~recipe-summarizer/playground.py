import os
import cohere
from . import db

co = cohere.Client(os.getenv("COHERE_API_KEY"))


def embed_title(title) -> list[float]:
    return co.embed(
        texts=[title],
        model="embed-english-v2.0",
    ).embeddings[0]


def summarize_recipe(text, store=False) -> str:
    response = co.summarize(
        text=f"Extract all the relevant steps to make the recipe, include no redundant information: {text}",
        length="long",
        format="bullets",
        model="summarize-xlarge",
        temperature=0.3,
    )

    if store:
        # Do we even need this summarize step? Probablly not, right?
        title = co.summarize(
            text=f"Generate a title for this recipe: {text}, one sentence",
            length="short",
        )
        # Create a title for the recipe and then store it in the database
        vectors = embed_title(title.summary)
        db.add_summary(title.summary, response.summary, vectors)

    return response


def find_summarized_recipe(title: str) -> str:
    # Embed the title and then fetch the summary from the database
    vector = embed_title(title)
    return db.fetch_summary(vector)
