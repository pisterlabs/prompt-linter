# Std. lib imports
import os
import pathlib
import uuid

# Third party imports
import cohere
import chromadb
from chromadb.config import Settings
from dotenv import dotenv_values

# Internal imports
from .enriched_text import EnrichedText
from .note import EvergreenNote


ENV_LOCATION = pathlib.Path(__file__).parent.parent.resolve() / ".env"
CONFIG: dict = dotenv_values(ENV_LOCATION)
KB_COLLECTION_NAME = "kb_collection"
TARGET_DISTANCE: float = 1.0

cohere_client = cohere.Client(CONFIG["COHERE_KEY"])
chroma_client = chromadb.Client(
    Settings(
        chroma_db_impl="duckdb+parquet", persist_directory=CONFIG["CHROMADB_DIRECTORY"]
    )
)
kb_collection = chroma_client.get_or_create_collection(name=KB_COLLECTION_NAME)


def ingest_article(text: str) -> EnrichedText:
    enriched_text = EnrichedText(full_text=text)
    return enriched_text


def summarize_article(text: EnrichedText) -> str:
    # TODO: use human language interface in runner to determine type of summary (bullets vs paragraph)?
    summarize_response = cohere_client.summarize(
        text=text.enriched_text, length="long", temperature=0.2
    )
    return summarize_response.summary


def get_main_ideas(text: EnrichedText) -> str:
    summarize_response = cohere_client.summarize(
        text=text.enriched_text,
        length="long",
        format="bullets",
        model="summarize-xlarge",
        additional_command="focus on getting main ideas and arguments",
        temperature=0.7,
    )
    return [
        main_idea[2:] for main_idea in summarize_response.summary.strip().split("\n")
    ]


def generate_outline(text: EnrichedText) -> str:
    prompt = f"read the following text bounded by %%: %%{text.enriched_text}%% For the previous text, please generate a numbered outline that follows precisely the inherent structure of the text and would allow a reader to quickly understand what the text is about. If there are obvious headers, use those in the outline."
    response = cohere_client.generate(
        model="command-xlarge-nightly",
        prompt=prompt,
        max_tokens=300,
        temperature=0.4,
        k=0,
        stop_sequences=[],
        return_likelihoods="NONE",
    )
    return response.generations[0].text


def generate_evergreen_note_text(main_idea: str, outline: str) -> str:
    prompt = f"""
    Assertion: {main_idea}
    Outline: {outline}
    Using the outline as context and any outside information you have available, expand the assertion text into an evergreen note.
    """
    short_prompt = f"""
    Assertion: {main_idea}
    Using any outside information you have available, expand the assertion text into an evergreen note.
    """
    try:
        response = cohere_client.generate(
            model="command-xlarge-nightly",
            prompt=prompt,
            max_tokens=754,
            temperature=1.2,
            k=0,
            stop_sequences=[],
            return_likelihoods="NONE",
        )
    except cohere.CohereError as e:
        if e.http_status / 100 == 4:
            response = cohere_client.generate(
                model="command-nightly",
                prompt=short_prompt,
                max_tokens=754,
                temperature=1.2,
                k=0,
                stop_sequences=[],
                return_likelihoods="NONE",
            )
    return response.generations[0].text.rstrip()


def _parse_evergreen_note(note: EvergreenNote):
    raise NotImplementedError


def load_knowledgebase():
    """
    In the future, this will parse a note and use tags and any other detected metadata as metadata in building the collection
    Only run this once, otherwise you'll create duplicates
    """
    kb_collection = chroma_client.get_or_create_collection(name=KB_COLLECTION_NAME)

    evergreens = [
        note_title[:-3] for note_title in os.listdir(CONFIG["EVERGREEN_NOTE_DIRECTORY"])
    ]

    documents = []
    ids = []
    for evergreen in evergreens:
        documents.append(evergreen)
        ids.append(str(uuid.uuid4()))

    kb_collection.add(documents=documents, ids=ids)


def delete_knowledgebase():
    """
    Use only in case of emergency, such as creating duplicates in your DB
    """
    kb_collection.delete()


def find_and_connect_related_notes(evergreen_note: EvergreenNote) -> list[str]:
    """
    Search against database for similar notes
    """
    query_results = kb_collection.query(
        query_texts=[evergreen_note.title, evergreen_note.text], n_results=5
    )

    all_docs = query_results["documents"][0]
    all_docs.extend(query_results["documents"][1])
    all_distances = query_results["distances"][0]
    all_distances.extend(query_results["distances"][1])
    indexes = [
        idx for idx, distance in enumerate(all_distances) if distance < TARGET_DISTANCE
    ]
    match_docs = list(set([all_docs[idx] for idx in indexes]))
    return match_docs
