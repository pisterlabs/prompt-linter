from pathlib import Path

from langchain.agents import Tool
from llama_index import (
    LLMPredictor,
    PromptHelper,
    ServiceContext,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.indices.composability import ComposableGraph
from llama_index.indices.keyword_table import SimpleKeywordTableIndex
from llama_index.indices.query.query_transform.base import DecomposeQueryTransform
from llama_index.query_engine.transform_query_engine import TransformQueryEngine
from llama_index.vector_stores import ChromaVectorStore

from servifai.llm.openai import OpenAILLM
from servifai.memory.chroma import ChromaDB


class VectorKnowledgeBase:
    def __init__(self, vector_indices, index_summaries, service_context):
        self._indices = vector_indices
        self._summaries = index_summaries
        self._service_context = service_context

    def as_tool(self, title):
        index = self._indices[title]
        summary = self._summaries[title]
        query_engine = index.as_query_engine(
            service_context=self._service_context, similarity_top_k=3
        )
        return Tool(
            name=f"Knowledge Vector Index {title}",
            func=lambda q: str(query_engine.query(q)),
            description=f"useful for when you want to answer queries just on {summary}",
            return_direct=True,
        )


class KnowledgeGraphs:
    def __init__(self, vector_indices, index_summaries, service_context, llm_predictor):
        self._indices = vector_indices
        self._summaries = index_summaries
        self._service_context = service_context
        self._llm_pred = llm_predictor
        self._graph = None
        self._custom_query_engines = {}
        self._query_engine = self._create_graph_qe()

    def _create_graph_qe(self):
        self._graph = ComposableGraph.from_indices(
            SimpleKeywordTableIndex,
            [index for _, index in self._indices.items()],
            [summary for _, summary in self._summaries.items()],
            max_keywords_per_chunk=50,
        )
        decompose_transform = DecomposeQueryTransform(self._llm_pred, verbose=True)

        for index in self._indices.values():
            query_engine = index.as_query_engine(service_context=self._service_context)
            transform_extra_info = {"index_summary": index.index_struct.summary}
            tranformed_query_engine = TransformQueryEngine(
                query_engine,
                decompose_transform,
                transform_metadata=transform_extra_info,
            )
            self._custom_query_engines[index.index_id] = tranformed_query_engine

        self._custom_query_engines[
            self._graph.root_index.index_id
        ] = self._graph.root_index.as_query_engine(
            retriever_mode="simple",
            response_mode="tree_summarize",
            service_context=self._service_context,
        )
        return self._graph.as_query_engine(
            custom_query_engines=self._custom_query_engines
        )

    def as_tool(self, about):
        return Tool(
            name="Knowledge Graph Index",
            func=lambda q: str(self._query_engine.query(q)),
            description=f"useful for when you want to answer queries that require comparing/contrasting or analyzing over multiple sources of {about}",
            return_direct=True,
        )


class KnowledgeBase:
    def __init__(self, vdb_dir, data_dir, about, text, llm):
        self.vdb_dir = vdb_dir
        self.data_dir = data_dir
        self.about = about
        self.max_input_size = int(text["max_input_size"])
        self.num_outputs = int(text["num_outputs"])
        self.max_chunk_overlap = float(text["max_chunk_overlap"])
        self.chunk_size_limit = int(text["chunk_size_limit"])
        self.llm = llm
        self.oaillm = OpenAILLM(llm, text)
        self.llm_predictor = LLMPredictor(llm=self.oaillm.model)
        self.vectorstore = self._initiate_vectorstore()
        self.service_context = self._initiate_contexts()[0]
        self.storage_context = self._initiate_contexts()[1]
        self.titles = []
        self.indices = self._create_indices()[0]
        self.index_summaries = self._create_indices()[1]
        self.qe_tools = self._get_query_engine_tools()

    def _initiate_vectorstore(self):
        vectordb = ChromaDB(self.vdb_dir, self.llm["org"])
        return ChromaVectorStore(chroma_collection=vectordb.collection)

    def _initiate_contexts(self):
        prompt_helper = PromptHelper(
            self.max_input_size,
            self.num_outputs,
            self.max_chunk_overlap,
            chunk_size_limit=self.chunk_size_limit,
        )
        self.service_context = ServiceContext.from_defaults(
            llm_predictor=self.llm_predictor, prompt_helper=prompt_helper
        )
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vectorstore
        )
        return self.service_context, self.storage_context

    def _create_indices(self):
        """
        Loads the data, chunks it, create embedding for each chunk
        and then stores the embedding to a vector database.

        Args:
            data_dir (str): the directory containing the data
        """
        ext = ".pdf"
        docs = {}
        indices = {}
        index_summaries = {}

        for dbb in Path(self.data_dir).glob(f"*{ext}"):
            print(dbb)
            title = dbb.stem
            self.titles.append(title)
            docs[title] = SimpleDirectoryReader(
                input_files=[str(dbb)],
                recursive=True,
                exclude_hidden=True,
                required_exts=[ext],
            ).load_data()
            indices[title] = VectorStoreIndex.from_documents(
                docs[title],
                service_context=self.service_context,
                storage_context=self.storage_context,
            )
            index_summaries[
                title
            ] = f"individual source on {self.about} for particular {' '.join(title.split('-'))}"
        return indices, index_summaries

    def _get_query_engine_tools(self):
        qe_tools = []
        for title in self.titles:
            vectorkb_tool = VectorKnowledgeBase(
                self.indices, self.index_summaries, self.service_context
            ).as_tool(title)
            qe_tools.append(vectorkb_tool)

        kg_tool = KnowledgeGraphs(
            self.indices, self.index_summaries, self.service_context, self.llm_predictor
        ).as_tool(self.about)
        qe_tools.append(kg_tool)
        return qe_tools

    def as_tool(self):
        return self.qe_tools
