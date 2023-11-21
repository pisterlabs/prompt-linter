import datetime
import logging
import re
import os

from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.docstore.document import Document
from langchain.llms import OpenAIChat
from langchain.prompts import PromptTemplate
from scrapy.exceptions import DropItem

from langsearch.pipelines.common.index import BaseSimpleIndexPipeline
from langsearch.pipelines.common.mixins.summary import RecursiveReduceSummaryMixin
from langsearch.pipelines.common.mixins.weaviatedb import WeaviateMixin
from langsearch.utils import openai_length_function


logger = logging.getLogger(__name__)


PROMPT_TEMPLATE = """You are trying to find links that might contain the answer to the question: {question}

You have a few links, but you can't view all the information contained under the link. You only have access to a concise and incomplete summary of the information contained in those links. Therefore, the summaries may not contain the answer to the question directly. The links themselves contain a lot more information than the summary. You need to decide which links to investigate further, i.e view their full content.

{context}

For which links would you fetch the full content to see if they contain the answer to the following question: {question}

Remember, the summaries may not contain the answer to the question directly, because they are incomplete. The links themselves contain a lot more information than the summary.

Please provide a list of all those links to investigate further.

List of links:
"""

PROMPT = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["question", "context"])


class BaseSummaryIndexPipeline(BaseSimpleIndexPipeline):
    INPUTS = {
        "url": "url",
        "sections": "sections",
        "changed": "changed",
        "summarizer_input": "summarizer_input",
    }
    SUMMARY_CLASS_SCHEMA = {
        "class": "Summary",
        "vectorizer": "text2vec-transformers",
        "moduleConfig": {
            "text2vec-transformers": {
                "vectorizeClassName": False,
            }
        },
        "properties": [
            {
                "name": "url",
                "description": "The URL of the document",
                "dataType": ["string"],
                "moduleConfig": {
                    "text2vec-transformers": {
                        "skip": True,
                    }
                },
            },
            {
                "name": "summary",
                "description": "Summary of the page",
                "dataType": ["text"],
                "moduleConfig": {
                    "text2vec-transformers": {
                        "skip": False,
                        "vectorizePropertyName": False
                    }
                },
            },
            {
                "name": "last_seen",
                "description": "When this URL was last seen in the crawling process",
                "dataType": ["date"],
                "moduleConfig": {
                    "text2vec-transformers": {
                        "skip": True,
                    }
                },
            }
        ],
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        summary_class_schema = self.__class__.get_setting_from_partial_key(os.environ, "SUMMARY_CLASS_SCHEMA")
        if isinstance(summary_class_schema, str):
            summary_class_schema = self.get_params_from_file(summary_class_schema)
        self.summary_class_schema = summary_class_schema
        self.summary_class_name = self.summary_class_schema["class"]
        if not self.weaviate.class_exists(self.summary_class_name):
            self.weaviate.create_class(self.summary_class_schema)

    def get_all_summaries_for_url(self):
        where_filter = {
            "path": ["url"],
            "operator": "Equal",
            "valueString": self.url
        }
        result = (
            self.weaviate.client.query
            .get(self.summary_class_name)
            .with_additional(["id"])
            .with_where(where_filter)
            .do()
        )
        return result["data"]["Get"][self.summary_class_name]

    def apply(self, item, spider):
        if not hasattr(self, "url"):
            return item
        if not hasattr(self, "summarizer_input"):
            return item
        try:
            data = self.get_all_summaries_for_url()
            count = len(data)
            if count > 1:
                message = (
                        f"Found {count} data obj for unique property URL {self.url}"
                        f"in class {self.summary_class_name}"
                )
                logger.warning(message)
                raise RuntimeError(message)
            elif count == 0:
                self.weaviate.client.data_object.create(
                    class_name=self.summary_class_name,
                    data_object={
                        "url": self.url,
                        "summary": self.summarize(**self.summarizer_input),
                        "last_seen": datetime.datetime.now(datetime.timezone.utc).isoformat()
                    }
                )
            else:
                if hasattr(self, "changed") and not self.changed:
                    self.weaviate.update_property_with_current_datetime(
                        class_name=self.summary_class_name,
                        where_filter={
                            "path": ["url"],
                            "operator": "Equal",
                            "valueString": self.url
                        },
                        property_name="last_seen"
                    )
                else:
                    unique_id = data[0]["_additional"]["id"]
                    self.weaviate.client.data_object.delete(
                        class_name=self.summary_class_name,
                        uuid=unique_id
                    )
                    self.weaviate.client.data_object.create(
                        class_name=self.summary_class_name,
                        data_object={
                            "url": self.url,
                            "summary": self.summarize(**self.summarizer_input),
                            "last_seen": datetime.datetime.now(datetime.timezone.utc).isoformat()
                        }
                    )
        except:
            message = f"Error while storing summary for item with URL {self.url}"
            logger.exception(message)
            raise DropItem(message)
        else:
            super().apply(item, spider)

    @staticmethod
    def extract_links(text):
        return re.findall("(?:http|file)s?://[^\s]+", text)

    def get_similar_summaries(self, text, **kwargs):
        result = self.weaviate.get_similar(self.summary_class_name, text, ["url", "summary", "last_seen"], **kwargs)
        return [Document(page_content=item["summary"], metadata={"source": item["url"]})
                for item in result
                ]

    def get_similar_summaries_using_llm(
            self,
            text,
            llm=OpenAIChat(temperature=0, max_tokens=256),
            max_tokens=256,  # This should match the max_tokens of the LLM,
            max_context_size=4096,
            length_function=openai_length_function,
            prompt=PROMPT,
            document_variable_name="context",
            document_prompt_template="LINK: {source}\nSUMMARY: {page_content}",
            **kwargs
            ):
        prompt_length = length_function(prompt.format(**{input_var: "" for input_var in prompt.input_variables}))
        document_prompt = PromptTemplate(
            input_variables=["page_content", "source"],
            template=document_prompt_template
        )
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        stuff_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_prompt=document_prompt,
            document_variable_name=document_variable_name
        )

        summaries = self.get_similar_summaries(text, **kwargs)
        token_count = 0
        remaining = max_context_size - prompt_length - max_tokens - 50  # -50 for safety
        matching_summaries = []
        docs = []
        for summary in summaries:
            token_count += length_function(f"LINK: {summary.metadata['source']}\nSUMMARY: {summary.page_content}\n\n")
            if token_count > remaining:
                answer = stuff_chain.run(input_documents=docs, question=text)
                logger.debug(f"Summary select answer: {answer}")
                extracted_links = self.extract_links(answer)
                matching_summaries.extend([doc for doc in docs if doc.metadata["source"] in extracted_links])
                token_count = 0
                docs = [summary]
            else:
                docs.append(summary)
        answer = stuff_chain.run(input_documents=docs, question=text)
        logger.debug(f"Summary select answer: {answer}")
        extracted_links = self.extract_links(answer)
        matching_summaries.extend([doc for doc in docs if doc.metadata["source"] in extracted_links])
        return matching_summaries


class SummaryIndexPipeline(BaseSummaryIndexPipeline, WeaviateMixin, RecursiveReduceSummaryMixin):
    pass
