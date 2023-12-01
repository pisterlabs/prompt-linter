import requests
import os

from llama_index.llms import OpenAI
from llama_index.graph_stores import SimpleGraphStore

from src.graphindex.common.config import (
    SCHEMA_ORG_URI,
    SCHEMA_ORG_LOCAL_PATH_SUBGRAPHS,
    SCHEMA_ORG_LOCAL_PATH,
    SCHEMA_ORG_INDEX_LOCAL_PATH,
    SCHEMA_FILE_NAME
)

from llama_index import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    SimpleDirectoryReader,
    KnowledgeGraphIndex,
    ServiceContext
)

from src.graphindex.common.utils import (
    create_graph_from_jsonld,
    save_subjects_to_files,
    extract_subgraph_for_each_subject
)

from src.graphindex.common.enumerations import IndexType


class OntologyIndex:
    def __init__(
            self,
            index_type: IndexType,
            source_dir='./schemas',
            output_dir='./index',
            use_schema_org: bool = False,
            ontology_version: str = None
    ) -> None:
        self.uri = SCHEMA_ORG_URI.format(ontology_version=ontology_version)
        self.index_type = index_type
        self.schema_file_name = SCHEMA_FILE_NAME

        if use_schema_org:
            self.schema_local_path = SCHEMA_ORG_LOCAL_PATH.format(
                src_dir=source_dir,
                ontology_version=ontology_version
            )
            self.local_index = SCHEMA_ORG_INDEX_LOCAL_PATH.format(
                output_dir=output_dir,
                ontology_version=ontology_version,
                index_type=str.lower(self.index_type.name)
            )
            self.subgraphs_path = SCHEMA_ORG_LOCAL_PATH_SUBGRAPHS.format(
                src_dir=source_dir,
                ontology_version=ontology_version
            )

        else:
            self.schema_local_path = source_dir
            self.local_index = output_dir
            self.subgraphs_path = f"{source_dir}/subgraphs"

        self.index = None
        self.ontology_version = ontology_version
        self.use_schema_org= use_schema_org

    def _is_local_ontology(self):
        local_path = self.schema_local_path

        if os.path.isdir(local_path):
            return True

        os.makedirs(local_path)
        return False

    def _load_local_index(self):
        local_index = self.local_index

        if os.path.isdir(local_index):
            storage_context = StorageContext.from_defaults(persist_dir=local_index)
            return load_index_from_storage(storage_context)

        os.makedirs(local_index)

        return None

    def _load_schema_org_ontology(self):
        if not self._is_local_ontology():
            release_uri = self.uri

            response = requests.get(release_uri)

            if response.status_code != 200:
                raise Exception(f"Failed to fetch schema.org version {self.ontology_version}. \
                            Status code: {response.status_code}")

            result = response.json()

            graph = create_graph_from_jsonld(result)
            subjects_with_props = extract_subgraph_for_each_subject(graph)
            save_subjects_to_files(
                subjects_with_props,
                self.subgraphs_path
            )

    def _transform_graph_to_vector_store(self):
        try:
            index = self._load_local_index()
        except Exception as err:
            raise Exception("Could not read local index.")

        if not index:
            schema_path = self.subgraphs_path
            index_path = self.local_index

            documents = SimpleDirectoryReader(schema_path).load_data()

            if self.index_type == IndexType.VECTOR:
                index = VectorStoreIndex.from_documents(documents)

            elif self.index_type == IndexType.KNOWLEDGE_GRAPH:
                graph_store = SimpleGraphStore()
                storage_context = StorageContext.from_defaults(graph_store=graph_store)

                llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
                service_context = ServiceContext.from_defaults(llm=llm, chunk_size=512)

                # NOTE: can take a while!
                index = KnowledgeGraphIndex.from_documents(
                    documents,
                    max_triplets_per_chunk=3,
                    storage_context=storage_context,
                    service_context=service_context,
                )
            else:
                raise NotImplementedError()

            index.storage_context.persist(persist_dir=index_path)

        self.index = index

    def create_index_from_ontology_version(self):
        if self.use_schema_org:
            self._load_schema_org_ontology()
        self._transform_graph_to_vector_store()

    def get_index(self):
        return self.index
