import os
from gptcache import cache, Config
from gptcache.processor.pre import concat_all_queries
from gptcache.manager import get_data_manager
from gptcache.manager.scalar_data.sql_storage import SQLStorage
from gptcache.manager.vector_data.faiss import Faiss
from gptcache.embedding.openai import OpenAI
from gptcache.adapter import openai
from gptcache.utils.log import gptcache_log

from packages.medagogic_sim.logger.logger import get_logger, logging

logger = get_logger(level=logging.DEBUG)

import dotenv


def configure_cached_openai(directory=None):
    """
    Also import openai from this file or else
    """

    if cache.has_init:
        return

    if directory is None:
        directory = os.path.dirname(os.path.abspath(__file__))

    db_path = os.path.join(directory, "gptcache.db")
    faiss_index_path = os.path.join(directory, "gptcache.faiss.index")

    dotenv.load_dotenv()

    embedder = OpenAI()
    scalar_storage = SQLStorage(url=f"sqlite:///{db_path}")
    vector_storage = Faiss(dimension=embedder.dimension, index_file_path=faiss_index_path, top_k=1)
    data_manager = get_data_manager(scalar_storage, vector_storage)

    config = Config(
        skip_list=[],
        context_len=999999
    )

    def pre_embedding_func(queries, *args, **kwargs):
        # print(queries, args, kwargs)
        r = concat_all_queries(queries, *args, **kwargs)
        # print(r)
        return r

    cache.init(
        config=config,
        pre_embedding_func=pre_embedding_func,
        embedding_func=embedder.to_embeddings,
        data_manager=data_manager,
    )
    cache.set_openai_key()

    def hit_cache_callback(data):
        time, id = data
        logger.debug(f"Hit cache id {id} ({time}s)")

    data_manager.hit_cache_callback = hit_cache_callback

    def cache_enable_func(*args, **kwargs):
        return True

    cache.cache_enable_func = cache_enable_func


