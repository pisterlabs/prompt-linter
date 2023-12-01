from langchain.vectorstores import PGVector
from langchain.embeddings import CohereEmbeddings
from typing import List, Iterable, Optional, Any
from langchain.output_parsers.openai_functions import PydanticOutputFunctionsParser
import demjson3 as demjson
import copy


def clean_fnc_call(json_str):
    """Parse and re-encode the JSON string using demjson."""
    decoded_json = demjson.decode(json_str, strict=False)["output"]
    corrected_json_str = demjson.encode(decoded_json)
    return corrected_json_str


class CustomFixParser(PydanticOutputFunctionsParser):
    """Custom output parser."""

    def parse_result(self, result):
        generation = result[0]
        message = generation.message
        func_call = copy.deepcopy(message.additional_kwargs["function_call"])
        _result = func_call["arguments"]

        pydantic_args = self.pydantic_schema.parse_raw(clean_fnc_call(_result))
        return pydantic_args


class NewCohereEmbeddings(CohereEmbeddings):
    def embed_documents(self, texts: List[str], input_type: str) -> List[List[float]]:
        embeddings = self.client.embed(
            model=self.model, texts=texts, truncate=self.truncate, input_type=input_type
        ).embeddings
        return [list(map(float, e)) for e in embeddings]

    async def aembed_documents(
        self, texts: List[str], input_type: str
    ) -> List[List[float]]:
        embeddings = await self.async_client.embed(
            model=self.model, texts=texts, truncate=self.truncate, input_type=input_type
        )
        return [list(map(float, e)) for e in embeddings.embeddings]


class NewPGVector(PGVector):
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            kwargs: vectorstore specific parameters

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        embeddings = self.embedding_function.embed_documents(list(texts), input_type="search_document")
        return self.add_embeddings(
            texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids, **kwargs
        )