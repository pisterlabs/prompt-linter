import os
import textwrap

import numpy as np
import openai
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from sklearn.cluster import KMeans
from tenacity import stop_after_attempt  # for exponential backoff
from tenacity import retry, wait_random_exponential

DEFAULT_PROMPT = (
    "Summarize this Youtube video chapter. Always start with a topical sentence: "
)
CHAPTER_TITLE = "Give a title to this video chapter based on the transcript: "

title_template = "Give a title to this text summary: {text}"
TITLE_PROMPT = PromptTemplate(template=title_template, input_variables=["text"])

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_embeddings(text_chunks, openai_api_key, model="text-embedding-ada-002"):
    data = openai.Embedding.create(
        input=text_chunks, model=model, openai_api_key=openai_api_key
    )["data"]
    embeddings = [item["embedding"] for item in data]
    return np.array(embeddings)


def text_from_file(text_path):
    in_text = ""
    with open(text_path, "r", encoding="utf-8") as text_file:
        for line in text_file:
            in_text += line
    return in_text


def get_chunks(timestamped_transcripts, chunk_lines):
    chunks = []
    current_chunk = []
    for line in timestamped_transcripts:
        current_chunk.append(line)
        if len(current_chunk) == chunk_lines:
            chunks.append("\n".join(current_chunk))
            current_chunk = []

    if len(current_chunk) > 0:
        chunks.append("\n".join(current_chunk))

    return chunks


def align_chapters(timestamped_transcript, yt_chapters):
    timestamped_transcripts = timestamped_transcript.strip().split("\n")

    chapters = []
    chapter_text = ""
    chapter_start_time = 0.0
    prev_end_time = 0.0
    chapter_index = 0
    for idx, trn in enumerate(timestamped_transcripts):
        trn_start_time = float(trn.split()[0])
        trn_end_time = float(trn.split()[1])
        trn_text = " ".join(trn.split()[2:])

        if idx == 0:
            chapter_start_time = trn_start_time

        next_index = min(chapter_index + 1, len(yt_chapters) - 1)
        if trn_start_time >= yt_chapters[next_index]["start_time"]:
            if len(chapters) == len(yt_chapters):
                chapter_text += f"{trn_text}\n"
            else:
                chapters.append(
                    {
                        "text": chapter_text,
                        "start_time": chapter_start_time,
                        "end_time": prev_end_time,
                        "title": yt_chapters[chapter_index]["title"],
                    }
                )
                chapter_text = trn_text
                chapter_start_time = trn_start_time
                chapter_index += 1
        else:
            chapter_text += f"{trn_text}\n"
        prev_end_time = trn_end_time

    if len(chapters) == len(yt_chapters):
        chapter_index = len(yt_chapters) - 1
        chapters[chapter_index]["text"] += chapter_text
        chapters[chapter_index]["end_time"] = prev_end_time
    return chapters


def get_automatic_chapters(
    timestamped_transcript, openai_api_key, chunk_lines=5, num_clusters=3
):
    timestamped_transcripts = [
        timestamped_line
        for timestamped_line in timestamped_transcript.split("\n")
        if len(timestamped_line.strip()) > 0
    ]

    # Split into chunks
    text_chunks = get_chunks(timestamped_transcripts, chunk_lines)
    embeddings = get_embeddings(text_chunks, openai_api_key)

    # Creating and fitting the K-means model
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(embeddings)

    # Getting the cluster labels
    cluster_labels = kmeans.labels_

    current_label = -1
    current_text = ""
    chapters = []
    for idx, (text_chunk, label) in enumerate(zip(text_chunks, cluster_labels)):
        start_time, end_time = get_chunk_timestamps(text_chunk)

        if idx == 0:
            chapter_start_time = start_time

        if label != current_label and current_label != -1:
            chapters.append(
                {
                    "text": current_text,
                    "start_time": chapter_start_time,
                    "end_time": prev_end_time,
                    "title": "",
                }
            )
            current_text = ""
            chapter_start_time = start_time

        current_label = label
        current_text += get_chunk_text(text_chunk)
        prev_end_time = end_time
    if len(current_text) > 0:
        chapters.append(
            {
                "text": current_text,
                "start_time": chapter_start_time,
                "end_time": prev_end_time,
                "title": "",
            }
        )
    return chapters


def get_chunk_timestamps(chunk):
    start_time = float(chunk.strip().split("\n")[0].split()[0])
    end_time = float(chunk.strip().split("\n")[-1].split()[1])
    return start_time, end_time


def get_chunk_text(chunk):
    chunk_text = ""
    for chunk_line in chunk.strip().split("\n"):
        chunk_text += " ".join(chunk_line.split()[2:])
    return chunk_text


def summarize_chapters(chapters, openai_api_key):
    llm = OpenAI(temperature=0.9, openai_api_key=openai_api_key)
    chapter_docs = [Document(page_content=chapter["text"]) for chapter in chapters]

    summary_chain = load_summarize_chain(
        llm, chain_type="map_reduce", return_intermediate_steps=True
    )
    summaries = summary_chain(
        {"input_documents": chapter_docs}, return_only_outputs=True
    )

    summary_docs = [
        Document(page_content=summary) for summary in summaries["intermediate_steps"]
    ]

    title_chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        return_intermediate_steps=True,
        map_prompt=TITLE_PROMPT,
    )
    titles = title_chain({"input_documents": summary_docs}, return_only_outputs=True)

    summarized_chapters = []
    for chapter, chapter_summary, chapter_title in zip(
        chapters, summaries["intermediate_steps"], titles["intermediate_steps"]
    ):
        if len(chapter["title"]) > 0:
            chapter_title = chapter["title"]
        summarized_chapters.append(
            {
                "start": chapter["start_time"],
                "end": chapter["end_time"],
                "text": chapter["text"],
                "title": chapter_title.strip(),
                "summary": chapter_summary.strip(),
            }
        )
    return summarized_chapters, summaries["output_text"]
