import asyncio
import datetime
import re
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
)

import numpy as np
import spacy
from elasticsearch import Elasticsearch
from langchain.callbacks.base import Callbacks
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.chains import RetrievalQA
from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain.chat_models import ChatVertexAI, ChatOpenAI
from langchain.docstore.document import Document
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.output_parsers import NumberedListOutputParser
from langchain.prompts import PromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, \
    ChatPromptTemplate
from langchain.schema import BaseRetriever
from langchain.schema.embeddings import Embeddings
from langchain.vectorstores import ElasticsearchStore
from langchain.vectorstores.utils import maximal_marginal_relevance
from pydantic import BaseModel, Field
from spacy.tokens import Doc

try:
    import config
except ModuleNotFoundError:
    import summarizer.config as config


class ContextFusionQuery(BaseModel):
    rewritten_query: str = Field(description="The rewritten of the current query that's independent of the previous "
                                             "query and response")
    alternative_queries: List[str] = Field(description="Alternative queries based on the current query, the previous "
                                                       "query, and the previous response", default=list())


async def afind_top_topics(vector_db, query, now, delta, model, k=5):
    """
    async function that identifies the topics relevant to the query in a given time frame

    :param vector_db: the ElasticsearchStore that allows querying over the topics
    :param query: the query string
    :param now: the current time
    :param delta: datetime.timedelta determining how far back to go
    :param model: the topic model id identifying the model that generated the topics
    :param k: the number of topics to find. defaults to 5.
    :returns: list of Langchain Documents containing the top topics
    """
    start_date = now - delta
    search_args = [{"term": {"metadata.model": model}},
                   {"range": {"metadata.recency": {"gte": start_date.strftime("%Y-%m-%d")}}}]
    topic_results = await vector_db.asimilarity_search(query=query, k=k, filter=search_args)
    return topic_results


def find_top_topics(vector_db, query, now, delta, model, k=5):
    """
    function that identifies the topics relevant to the query in a given time frame

    :param vector_db: the ElasticsearchStore that allows querying over the topics
    :param query: the query string
    :param now: the current time
    :param delta: datetime.timedelta determining how far back to go
    :param model: the topic model id identifying the model that generated the topics
    :param k: the number of topics to find. defaults to 5.
    :returns: list of Langchain Documents containing the top topics.
    """

    start_date = now - delta
    search_args = [{"term": {"metadata.model": model}},
                   {"range": {"metadata.recency": {"gte": start_date.strftime("%Y-%m-%d")}}}]
    return vector_db.asimilarity_search(query, k=k, filter=search_args)


class ArticleChunkRetriever(BaseRetriever):
    """
    A LangChain retriever that retrieves article chunks from elastic search, and then augment it in a format that
    the model expects.
    """

    #: the ElasticsearchStore for the article chunks
    chunks_elasticstore: ElasticsearchStore
    #: the number of article chunks to retrieve from each topic
    chunk_k: int = config.ARTICLE_K
    #: the topic to retrieve from
    topic: int = -10
    #: a datetime.timedelta representing how far back to go
    time_delta: datetime.timedelta
    #: the current time
    now: datetime.datetime
    #: the topic model id
    model: str

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        start_date = self.now - self.time_delta
        search_args = [{"term": {"metadata.model": self.model}},
                       {"range": {"metadata.published_at": {"gt": start_date.strftime("%Y-%m-%d")}}}]
        if self.topic >= -1:
            search_args.append({"term": {"metadata.topic": self.topic}})
        results = self.chunks_elasticstore.max_marginal_relevance_search(query=query, k=self.chunk_k,
                                                                         filter=search_args)
        for doc in results:
            publish_str = f"Published: {doc.metadata['published_at'].strftime('%Y-%m-%d')}"
            doc.page_content = publish_str + "\n" + doc.page_content.strip()
        return results

    async def aget_relevant_documents(self, query: str, *, callbacks: Callbacks = None, tags: Optional[
        List[str]] = None, metadata: Optional[Dict[str, Any]] = None, run_name: Optional[str] = None, **kwargs: Any):
        start_date = self.now - self.time_delta
        search_args = [{"term": {"metadata.model": self.model}},
                       {"range": {"metadata.published_at": {"gt": start_date.strftime("%Y-%m-%d")}}}]
        if self.topic >= -1:
            search_args.append({"term": {"metadata.topic": self.topic}})
        results = await self.chunks_elasticstore.amax_marginal_relevance_search(query=query, k=self.chunk_k,
                                                                                filter=search_args)
        for doc in results:
            publish_str = f"Published: {doc.metadata['published_at']}"
            doc.page_content = publish_str + "\n" + doc.page_content.strip()
        return results


class ArticleChunkHybridRetriever(BaseRetriever):
    """
    A retriever that uses all available fields in the article index to query for chunks
    """

    #: the asynchronous elastic search client
    elastic_client: Elasticsearch
    #: spacy pipeline
    spacy_nlp: Any
    #: sentence transformer
    encoder: Embeddings
    #: index name
    index_name: str = "articles"
    #: the chunk field
    page_content: str = "chunk_text"
    #: text fields to query
    text_fields: List[str] = ["chunk_text", "topic_text", "metadata.title", "metadata.entities"]
    #: vector fields to query
    vector_field: str = "chunk_text_embedding"
    #: mmr lambda
    mmr_lambda = 0.8
    #: the number of chunks to fetch for MMR
    fetch_k: int = 50
    #: the number of candidates to filter
    candidate_k: int = 200
    #: the number of article chunks to retrieve from each topic
    chunk_k: int = config.ARTICLE_K
    #: the topic to retrieve from
    topic: int = -100
    #: a datetime.timedelta representing how far back to go
    time_delta: datetime.timedelta
    #: the current time
    now: datetime.datetime
    #: the topic model id
    model: str
    #: RRF Constant
    rrf_k: int = 20

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        results = self.max_marginal_relevance_search(query)
        for doc in results:
            publish_str = f"Published: {doc.metadata['published_at']}"
            doc.page_content = publish_str + "\n" + doc.page_content.strip()
        return results

    def max_marginal_relevance_search(self, query: str) -> List[Document]:
        # Embed the query
        query_embedding = self.encoder.embed_query(query)

        # Fetch the initial documents
        got_docs = self._search(
            query=query,
            query_vec=query_embedding
        )

        # Get the embeddings for the fetched documents
        got_embeddings = [doc.metadata[self.vector_field] for doc, _ in got_docs]

        # Select documents using maximal marginal relevance
        selected_indices = maximal_marginal_relevance(
            np.array(query_embedding), got_embeddings, lambda_mult=self.mmr_lambda, k=self.chunk_k
        )
        selected_docs = [got_docs[i][0] for i in selected_indices]

        return selected_docs

    def _search(self, query: str, query_vec: List[float]) -> List[Tuple[Document, float]]:
        fields = ["metadata", self.vector_field, self.page_content]
        # Perform the hybrid search on the Elasticsearch index and return the results.
        response = self.elastic_client.search(
            index=self.index_name,
            **self._build_query(query_vector=query_vec, query_text=query,
                                vector_query_field=self.vector_field,
                                text_fields=self.text_fields),
            size=self.fetch_k,
            source=fields,
        )

        docs_and_scores = []
        for hit in response["hits"]["hits"]:
            for field in fields:
                if field in hit["_source"] and field not in [
                    "metadata",
                    self.page_content,
                ]:
                    hit["_source"]["metadata"][field] = hit["_source"][field]

            docs_and_scores.append(
                (
                    Document(
                        page_content=hit["_source"].get(self.page_content, ""),
                        metadata=hit["_source"]["metadata"],
                    ),
                    hit["_score"],
                )
            )
        return docs_and_scores

    def _build_filter(self):
        start_date = self.now - self.time_delta
        search_args = [{"term": {"metadata.model": self.model}},
                       {"range": {"metadata.published_at": {"gt": start_date.strftime("%Y-%m-%d")}}}]
        if self.topic >= -1:
            search_args.append({"term": {"metadata.topic": self.topic}})
        return search_args

    def _build_knn_subquery(self, field, vector):
        return {
            "filter": self._build_filter(),
            "field": field,
            "query_vector": vector,
            "k": self.fetch_k,
            "num_candidates": self.candidate_k,
        }

    def _build_query(
            self,
            query_vector: List[float],
            query_text: str,
            vector_query_field: str,
            text_fields: List[str],
    ) -> Dict:
        text_doc: Doc = self.spacy_nlp(query_text)
        keywords = []
        for ent in text_doc.ents:
            keywords.append(ent.text.strip())

        if len(keywords) > 0:
            return {
                "query": {
                    "multi_match": {
                        "query": " ".join(keywords),
                        "type": "best_fields",
                        "fields": text_fields
                    }
                },
                "knn": self._build_knn_subquery(vector_query_field, query_vector),
                "rank": {
                    "rrf": {
                        "window_size": self.fetch_k,
                        "rank_constant": self.rrf_k
                    }
                }
            }
        else:
            return {
                "knn": self._build_knn_subquery(vector_query_field, query_vector)
            }


class RAGFusionRetriever(BaseRetriever):
    queries: List[str]
    base_retriever: BaseRetriever
    boost: float = 2.0
    default_rank: int = 100
    top: int = config.FUSION_CHUNKS
    k: int = 60

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        results = {}
        rankers = []
        for q in self.queries:
            result = self.base_retriever.get_relevant_documents(q, run_manager=run_manager)
            rankers.append(self._create_rank_function(result))
            for r in result:
                if r.page_content not in results:
                    results[r.page_content] = r

        ranked = self._rrf(rankers, results.values())
        return [d for _, d in sorted(ranked, key=lambda x: x[0])[:self.top]]

    async def aget_relevant_documents(self, query: str, *, callbacks: Callbacks = None, tags: Optional[
        List[str]] = None, metadata: Optional[Dict[str, Any]] = None, run_name: Optional[str] = None,
                                      **kwargs: Any):
        results = {}
        rankers = []
        query_coros = []
        for q in self.queries:
            query_coros.append(self.base_retriever.aget_relevant_documents(q, callbacks=callbacks, tags=tags,
                                                                           metadata=metadata,
                                                                           run_name=run_name, **kwargs))
        query_coros = await asyncio.gather(*query_coros)
        for result in query_coros:
            print(f"Got: {len(result)} raw results")
            rankers.append(self._create_rank_function(result))
            for r in result:
                if r.page_content not in results:
                    results[r.page_content] = r

        ranked = self._rrf(rankers, results.values())
        print(f"Ended with {len(ranked)} RRF fusion results")
        return [d for _, d in sorted(ranked, key=lambda x: x[0])[:self.top]]

    def _create_rank_function(self, documents: List[Document]):
        doc_map = {page.page_content: i + 1 for i, page in enumerate(documents)}

        def rank_function(document: Document):
            if document.page_content in doc_map:
                return doc_map[document.page_content]
            return self.default_rank

        return rank_function

    def _rrf(self, rankers: List[Callable], documents: Iterable[Document], boosted=0) -> List[
        Tuple[float, Document]]:
        results = []
        for doc in documents:
            weights = []
            for i, ranker in enumerate(rankers):
                weight = 1 / (self.k + ranker(doc))
                if i == boosted:
                    weight = weight * self.boost
                weights.append(weight)
            results.append((sum(weights), doc))
        return results


class ChainRetriever(BaseRetriever):
    """
    An adaptor that converts BaseRetrievalQA chains into retriever for other chains. It will call each
    given chain with the query, and then return the results from the chains as the retrieved documents. The
    metadata from the chains are preserved.
    """

    #: list of BaseRetrievalQA chains to adapt
    chains: List[BaseRetrievalQA]

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        result = []
        for chain in self.chains:
            process_result = chain(query)
            page_content = process_result["result"]
            meta_data = {}
            if "source_documents" in process_result:
                meta_data["source_documents"] = process_result["source_documents"]
            result.append(Document(page_content=page_content, metadata=meta_data))
        return result


def topic_aggregate_chain(model, retriever, **kwargs):
    """
    Creates a langchain that can be used to do QA.

    :param model: the langchain LLM to use
    :param retriever: the retriever to use
    :param kwargs: keyword arguments for from_chain_type for RetrievalQA
    """

    if config.QA_MODEL == "custom":
        chain_type_kwargs = {"prompt": PromptTemplate.from_template(config.QA_LLAMA_PROMPT)}
    else:
        chain_type_kwargs = {"prompt": PromptTemplate.from_template(config.QA_RESP_PROMPT)}
    final_chain = RetrievalQA.from_chain_type(llm=model, chain_type="stuff",
                                              retriever=retriever,
                                              chain_type_kwargs=chain_type_kwargs,
                                              return_source_documents=True,
                                              **kwargs)
    return final_chain


def create_keypoints_chain(chunk_db, topic, topic_model, model,
                           now: datetime.datetime, delta: datetime.timedelta, k=7):
    """
    Creates a langchain that can be used to do keypoint summaries by topic

    :param chunk_db: ElasticsearchStore that points to the article chunks
    :param topic: the topic number that the chain is responsible for
    :param topic_model: the topic model id number for the topic
    :param model: the langchain LLM to use
    :param now: the current time
    :param delta: how far to go back
    :param k: the number of article chunks to retrieve for summarization
    """
    if config.SUM_MODEL == "custom":
        chain_type_kwargs = {"prompt": PromptTemplate.from_template(config.TOPIC_SUM_MISTRAL_PROMPT)}
    else:
        chain_type_kwargs = {"prompt": PromptTemplate.from_template(config.TOPIC_SUM_GENERIC_PROMPT)}
    retriever = ArticleChunkRetriever(chunks_elasticstore=chunk_db, chunk_k=k, topic=topic,
                                      time_delta=delta, now=now, model=topic_model)
    chain = RetrievalQA.from_chain_type(llm=model, chain_type="stuff",
                                        retriever=retriever,
                                        return_source_documents=True,
                                        chain_type_kwargs=chain_type_kwargs)
    return chain


async def aget_summaries(query, topics, now, delta, topic_model, chunk_db, model, top_k=config.TOPIC_SUM_TOP_K,
                         chunk_k=config.TOPIC_SUM_CHUNKS):
    """
    async method for computing the key points on a topic by topic basis.

    :param query: the user query
    :param topics: a list of topics to compute summaries for
    :param now: the current time
    :param delta: how far back to go
    :param topic_model: the topic model number
    :param chunk_db: the ElasticsearchStore for the article chunks
    :param model: the langchain LLM to use
    :param top_k: the number of topic summaries to return
    :param chunk_k: the number of chunks per topic to use for summarization.
    """

    key_chains = [create_keypoints_chain(chunk_db, t, topic_model, model, now, delta, k=chunk_k) for t in topics]
    print(f"Computing summaries for {len(key_chains)} topics")
    tasks = [c.acall(query) for c in key_chains]
    inter_results = await asyncio.gather(*tasks)
    split_regex = re.compile(r"\n\*")

    results = []
    for r in inter_results:
        result_text = r["result"].strip()
        if "impossible" in result_text.lower():
            continue
        parts = split_regex.split(result_text)

        results.append({
            "title": parts[0],
            "keypoints": parts[1:] if len(parts) > 1 else [],
            "sources": r["source_documents"]
        })
        if len(results) >= top_k:
            break

    return results


if __name__ == "__main__":
    with open(config.ES_KEY_PATH, "r") as fp:
        es_key = fp.read().strip()
    with open(config.ES_CLOUD_ID_PATH, "r") as fp:
        es_id = fp.read().strip()
    elastic_client = Elasticsearch(cloud_id=es_id, api_key=es_key)
    nlp = spacy.load("en_core_web_sm", enable=["ner"])
    embedding = SentenceTransformerEmbeddings(model_name=config.FILTER_EMBEDDINGS, model_kwargs={'device': 'cpu'})
    now = datetime.datetime(year=2023, month=11, day=10)
    delta = datetime.timedelta(days=999)
    model = "32f3de19-18ef-4cfe-b65a-b7c13116075d"

    # Plain Elastic Search Retriever
    hybrid_retriver = ArticleChunkHybridRetriever(
        elastic_client=elastic_client,
        spacy_nlp=nlp,
        encoder=embedding,
        fetch_k=config.FUSION_SRC_CHUNKS,
        candidate_k=config.FUSION_SRC_CHUNKS * 3,
        chunk_k=config.FUSION_SRC_CHUNKS,
        mmr_lambda=0.9,
        time_delta=delta,
        now=now,
        model=model
    )

    query = "What are some key developments in AI?"
    print(f"Query: {query}")
    docs = hybrid_retriver.get_relevant_documents(query)
    print("Hybrid Elastic Search:")
    for d in docs[:config.ARTICLE_K]:
        print(d.metadata["title"])


    # Fusion Retriever
    def get_rewrite_llm(kind=config.REWRITE_MODEL, max_token=1024):
        """
        Gets a langchain LLM for injecting context into query.

        :param kind: the type of model. Currently it supports PaLM2 ("vertexai"), GPT-3.5 ("openai")
        :param max_token: the maximum number of tokens that can be emitted.
        """
        if kind == "vertexai":
            plan_llm = ChatVertexAI(
                project=config.GCP_PROJECT,
                temperature=0,
                model_name="chat-bison",
                max_output_tokens=max_token
            )
            return plan_llm
        elif kind == "openai":
            with open(config.OPENAI_API) as fp:
                key = fp.read()
            plan_llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=key,
                                  temperature=0, max_tokens=max_token)
            return plan_llm
        else:
            raise NotImplemented()


    # Set up LangChain
    output_parser = NumberedListOutputParser()
    rewrite_system = SystemMessagePromptTemplate.from_template(config.REWRITE_SYSTEM_PROMPT)
    rewrite_user = HumanMessagePromptTemplate.from_template(config.REWRITE_USER_PROMPT)
    rewrite_history = []
    rewrite_chat = ChatPromptTemplate.from_messages([rewrite_system, *rewrite_history, rewrite_user])
    fusion_system = SystemMessagePromptTemplate.from_template(config.FUSION_SYSTEM_PROMPT)
    fusion_user = HumanMessagePromptTemplate.from_template(config.FUSION_USER_PROMPT)
    fusion_chat = ChatPromptTemplate.from_messages([fusion_system, fusion_user])
    llm = get_rewrite_llm()
    chain = fusion_chat.partial(
        format_instructions=output_parser.get_format_instructions()) | llm | output_parser

    # Get Fusion queries
    print("Fusion Search:")
    augmented_queries = chain.invoke({"query": query})
    print(f"Additional Queries: {augmented_queries}")

    fus_retriever = RAGFusionRetriever(rewrite_llm=get_rewrite_llm(),
                                       base_retriever=hybrid_retriver,
                                       default_rank=1000,
                                       queries=[query] + list(augmented_queries))
    docs = fus_retriever.get_relevant_documents(query)
    for d in docs:
        print(d.metadata["title"])
