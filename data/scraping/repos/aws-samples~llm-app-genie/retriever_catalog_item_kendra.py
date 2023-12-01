""" Module that contains a class that represents a Kendra retriever catalog item. """
from dataclasses import dataclass, field
from typing import Any, List, Tuple, Union

from langchain.retrievers.kendra import AmazonKendraRetriever
from langchain.schema import BaseRetriever

from .retriever_catalog_item import RetrieverCatalogItem

import boto3
from botocore.config import Config


@dataclass
class KendraRetrieverItem(RetrieverCatalogItem):
    """Class that represents a Kendra retriever catalog item."""

    region: str
    """ AWS Region """
    index_id: str
    """ Kendra Index ID """

    top_k: int
    """ Number of documents to be retrieved """

    _data_sources: Any
    """ Kendra Data Sources in this index """

    _selected_data_sources: List[Tuple[str, Any]]
    """ Kendra Data Source IDs that the retriever is supposed to use at the moment. """

    def __init__(self, friendly_name, index_id, data_sources, region=None, top_k=3):
        super().__init__(friendly_name)
        self._data_sources = data_sources
        self._selected_data_sources = []
        self.index_id = index_id
        self.region = region
        self.top_k = top_k

    @property
    def available_filter_options(self) -> Union[List[Tuple[str, Any]], None]:
        return [(data_src["Name"], data_src) for data_src in self._data_sources]

    @property
    def current_filter(self) -> List[str]:
        return [
            (data_src["Name"], data_src) for data_src in self._selected_data_sources
        ]

    @current_filter.setter
    def current_filter(self, value: List[Tuple[str, Any]]):
        selected_and_part_of_index = list(
            filter(lambda x: x[1] in self._data_sources, value)
        )
        self._selected_data_sources = selected_and_part_of_index

    def get_instance(self) -> BaseRetriever:
        data_src_filters = [
            {
                "EqualsTo": {
                    "Key": "_data_source_id",
                    "Value": {"StringValue": src[1]["Id"]},
                }
            }
            for src in self._selected_data_sources
        ]

        attribute_filter = {"OrAllFilters": data_src_filters}
        filter = attribute_filter

        if len(self._selected_data_sources) == 1:
            # Add data source language filter if the language is not English
            language = self._selected_data_sources[0][1].get("LanguageCode", None)
            if language and language != "en":
                lang_filter = {
                    "EqualsTo": {
                        "Key": "_language_code",
                        "Value": {
                            "StringValue": language
                        }
                    }
                }

                filter = {"AndAllFilters": [attribute_filter, lang_filter]}

        return AmazonKendraRetriever(
            index_id=self.index_id,
            region_name=self.region,
            attribute_filter=filter,
            top_k=self.top_k
        )
