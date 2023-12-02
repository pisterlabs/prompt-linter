import json
import logging
from abc import ABC
from typing import List, Dict, Union, Any

import requests
from llama_index import ServiceContext, get_response_synthesizer, Prompt
from llama_index.indices.knowledge_graph import KGTableRetriever
from llama_index.indices.postprocessor import SimilarityPostprocessor
from llama_index.indices.vector_store import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine

from src.graphindex.common.enumerations import IndexType
from src.graphindex.ontology import OntologyIndex
from llama_index.llms import OpenAI
from src.graphindex.common.prompts import (
    SEMANTIC_MAPPING_PROMPT_COLUMNS,
    SEMANTIC_MAPPING_REMAP_WRONG_RESULTS_COLUMNS
)


class BaseMapper(ABC):

    def __init__(self, openai_model: str = 'gpt-3.5-turbo') -> None:
        super().__init__()
        self.openai_model = openai_model

    def map(
            self,
            columns: List[str] = None,
            project_id: str = None,
            schema_id: str = None,
            table_id: str = None,
    ):
        ...

    @staticmethod
    def _check_map_arguments(
            columns: Dict[str, Dict[str, Any]],
            project_id: str,
            schema_id: str,
            table_id: str,
    ):

        if columns and (project_id or schema_id or table_id):
            raise TypeError(
                "Cannot provide project_id, schema_id or table_id if the argument columns is provided"
            )

        elif project_id and (not schema_id or not table_id):
            raise TypeError(
                "If project_id is provided then schema_id and table_id must also be provided"
            )

        elif not columns and not project_id:
            raise TypeError(
                "Must provide one of columns or (project_id, schema_id, table_id)"
            )

        return True


class SemanticMapper(BaseMapper):
    def __init__(
            self,
            openai_model: str = 'gpt-3.5-turbo-16k',
            target_ontology: str = 'schema.org',
            ontology_version: str = '22.0',
            ontology_source_dir: str = './schemas',
            index_output_dir: str = './indices',
            index_type: IndexType = IndexType.VECTOR,
            top_k: int = 2,
            similarity_cutoff: float = 0.7,
            temperature: float = 0.0,
            top_p: float = 0.01
    ) -> None:

        super().__init__(openai_model)
        self.target_ontology = target_ontology
        self.ontology_version = ontology_version
        self.top_k = top_k
        self.similarity_cutoff = similarity_cutoff
        self.temperature = temperature
        self.top_p = top_p
        self.ontology_source_dir = ontology_source_dir
        self.index_output_dir = index_output_dir
        self.index_type = index_type

        if target_ontology == 'schema.org':
            use_schema_org = True
        else:
            use_schema_org = False

        self.index = OntologyIndex(
            index_type=index_type,
            source_dir=ontology_source_dir,
            output_dir=index_output_dir,
            use_schema_org=use_schema_org,
            ontology_version=ontology_version
        )
        self.index.create_index_from_ontology_version()

        logging.info("Loaded index")

    def _llm_get_service_context(self, openai_model: str = None):
        if not openai_model:
            openai_model = self.openai_model
        llm = OpenAI(temperature=self.temperature, top_p=self.top_p, model=openai_model)
        return ServiceContext.from_defaults(llm=llm)

    def _postprocess_results(self, results, remap_llm: Union[str, None]):

        logging.info(results)

        try:
            results = json.loads(results, strict=False)
            for mapping in results['mappingResult']:
                term_uri = mapping['targetTermUri']
                if term_uri:
                    response = requests.get(term_uri)
                    logging.info(
                        f"Request to generated uri: {term_uri} responded with status code: {response.status_code}")
                    if response.status_code != 200:
                        mapping['targetTermUri'] = None
                        mapping['certainty'] = 'LOW'
                        mapping['reasoning'] = None
        except json.decoder.JSONDecodeError as e:
            raise Exception("Could not decode LLM generated results. Output structure doesn't conform to target.")

        if remap_llm:
            try:
                return self._map_unidentified_answers(results, remap_llm)
            except Exception as err:
                return results
        else:
            return results

    def _map_unidentified_answers(self, remap_answers: List[Dict[str, str]], llm):
        mapping = self.get_similar_terms_from_ontology_version(
            Prompt(SEMANTIC_MAPPING_REMAP_WRONG_RESULTS_COLUMNS),
            remap_answers,
            openai_model=llm
        )
        return self._postprocess_results(mapping.response, remap_llm=None)

    def map(
            self,
            columns: Union[Dict[str, Dict[str, Any]]] = None,
            description: str = None,
            project_id: str = None,
            schema_id: str = None,
            table_id: str = None,
            check_answers_llm: str = None,
    ):
        self._check_map_arguments(columns, project_id, schema_id, table_id)

        if project_id:
            raise NotImplementedError()

        else:
            mapping = self.get_similar_terms_from_ontology_version(
                Prompt(SEMANTIC_MAPPING_PROMPT_COLUMNS),
                columns,
                description
            )
            return self._postprocess_results(mapping.response, remap_llm=check_answers_llm)

    def get_similar_terms_from_ontology_version(
            self,
            query: Prompt,
            columns: Union[Dict[str, Dict[str, Any]], List[Dict[str, str]]],
            description: str = None,
            openai_model: str = None,
    ):

        if self.index_type == IndexType.VECTOR:
            retriever = VectorIndexRetriever(
                index=self.index.get_index(),
                similarity_top_k=self.top_k,
            )
        elif self.index_type == IndexType.KNOWLEDGE_GRAPH:
            retriever = KGTableRetriever(
                index=self.index.get_index(),
                similarity_top_k=self.top_k
            )
        else:
            raise NotImplementedError()

        # configure response synthesizer
        context = self._llm_get_service_context(openai_model)

        response_synthesizer = get_response_synthesizer(
            service_context=context,
            text_qa_template=query,
        )

        # assemble query engine
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=self.similarity_cutoff)
            ],
        )

        user_input = json.dumps(columns)
        if description:
            user_input += f"\n###\n# Description:\n{description}"

        return query_engine.query(user_input)
