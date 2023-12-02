import os
from loguru import logger
from lcserve import serving
import pickle
from langchain_utils import (
    get_summary,
    get_qa_with_sources,
    get_in_context_search,
    get_summarized_fact_check,
)
from data_utils import get_youtube_transcript, read_data_pickle
from typing import Dict, Union


allin_youtube_episodes_df = read_data_pickle("../data/allin_youtube_episodes_df.pkl")
logger.info(f"allin_youtube_episodes_df: {allin_youtube_episodes_df.columns}...")
allin_faiss_index = read_data_pickle("../data/allin_faiss_index.pkl")
try:
    cache = read_data_pickle("../data/summary_cache.pkl")
except:
    cache = {}


@serving
def get_summarized_topics(videoid: str, **kwargs) -> str:
    if videoid in cache:
        return cache[videoid]

    transcript = get_youtube_transcript(videoid)
    logger.info(f"Transcript: {transcript[:100]}...")
    if transcript:
        topic_summary = get_summary(transcript)
        cache[videoid] = topic_summary
        with open("../data/summary_cache.pkl", "wb") as f:
            pickle.dump(cache, f)
        return topic_summary
    else:
        return ""


@serving
def get_qa_search(querytext: str, **kwargs) -> Dict:
    answer = get_qa_with_sources(querytext, allin_faiss_index)
    return answer


@serving
def get_context_search(timestamp: Union[float, int], videoid: str, **kwargs) -> Dict:
    answer = get_in_context_search(
        timestamp, videoid, allin_youtube_episodes_df, allin_faiss_index
    )
    return answer


@serving
def get_fact_check(query: str, **kwargs) -> str:
    answer = get_summarized_fact_check(query)
    return answer
