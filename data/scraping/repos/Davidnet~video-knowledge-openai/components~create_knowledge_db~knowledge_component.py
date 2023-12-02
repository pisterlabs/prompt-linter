"""
Module that contains the process that creates the knowledge database.
"""
from kfp import dsl
from kfp.dsl import Artifact, Input

# pylint: disable=import-outside-toplevel


@dsl.component(
    target_image="us-central1-docker.pkg.dev/mlops-explorations/yt-whisper-images/create-knowledge-db:3.0",
    base_image="python:3.10",
    packages_to_install=["pinecone-client==2.2.1", "openai==0.27.4", "tenacity==8.2.2"],
)
def create_knowledge_db(transcriptions: Input[Artifact], window: int, stride: int):
    """
    Component that creates the knowledge database.
    param: transcriptions: Artifact that contains the transcriptions.
    """
    import json
    import os
    import shutil
    import tempfile
    from pathlib import Path

    import openai
    import pinecone
    from tenacity import (retry, stop_after_attempt,  # for exponential backoff
                          wait_random_exponential)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _embed_with_backoff(**kwargs):
        return openai.Embedding.create(**kwargs)

    openai.api_key = os.environ["OPENAI_API_KEY"]
    embed_model = "text-embedding-ada-002"

    stage_dir = Path(tempfile.mkdtemp())

    shutil.unpack_archive(transcriptions.path, extract_dir=stage_dir)
    data = []
    for json_file_lines in stage_dir.glob("*.jsonl"):
        with open(json_file_lines, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))

    window  # number of sentences to combine
    stride  # number of sentences to 'stride' over, used to create overlap

    new_data = []
    for i in range(0, len(data), stride):
        i_end = min(len(data) - 1, i + window)
        if data[i]["title"] != data[i_end]["title"]:
            # in this case we skip this entry as we have start/end of two videos
            continue
        text = " ".join([d["text"] for d in data[i:i_end]])
        # create the new merged dataset
        new_data.append(
            {
                "start": data[i]["start"],
                "end": data[i_end]["end"],
                "title": data[i]["title"],
                "text": text,
                "id": data[i]["video_id"],
                "url": data[i]["url"],
            }
        )

    index_name = "youtube-transcriptions"
    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        environment="us-west1-gcp",  # may be different, check at app.pinecone.io
    )
    # check if index already exists (it shouldn't if this is first time)
    if index_name not in pinecone.list_indexes():
        # if does not exist, create index
        pinecone.create_index(
            index_name,
            dimension=1536,  # OpenAI embedding dimension
            metric="cosine",
            # metadata_config={"indexed": ["channel_id", "published"]},
        )
    # connect to index
    index = pinecone.Index(index_name)
    print(index.describe_index_stats())

    batch_size = 100  # how many embeddings we create and insert at once
    for i in range(0, len(new_data), batch_size):
        # find end of batch
        i_end = min(len(new_data), i + batch_size)
        meta_batch = new_data[i:i_end]
        # get ids
        ids_batch = [x["id"] for x in meta_batch]
        # get texts to encode
        texts = [x["text"] for x in meta_batch]
        # create embeddings (try-except added to avoid RateLimitError)
        res = _embed_with_backoff(input=texts, engine=embed_model)
        embeds = [record["embedding"] for record in res["data"]]
        # cleanup metadata
        meta_batch = [
            {
                "start": x["start"],
                "end": x["end"],
                "title": x["title"],
                "text": x["text"],
                "url": x["url"],
            }
            for x in meta_batch
        ]
        to_upsert = list(zip(ids_batch, embeds, meta_batch))
        # upsert to Pinecone
        index.upsert(vectors=to_upsert)
