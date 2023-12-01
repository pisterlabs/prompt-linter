from typing import Optional

import abc

from bs4 import BeautifulSoup, Comment, NavigableString
from halo import Halo
from langchain.chat_models.base import BaseChatModel
from langchain.text_splitter import TextSplitter
from pydantic import BaseModel

from chatflock.parsing_utils import string_output_to_pydantic
from chatflock.structured_string import Section, StructuredString

from ..participants.langchain import LangChainBasedAIChatParticipant
from ..use_cases.request_response import get_response
from .errors import NonTransientHTTPError, TransientHTTPError
from .page_retrievers import PageRetriever


def clean_html(content):
    # Parse the HTML content
    soup = BeautifulSoup(content, "html.parser")

    # Remove non-visible tags
    for invisible_elem in soup(["style", "script", "meta", "[document]", "head", "title"]):
        invisible_elem.extract()

    # Remove comment nodes
    for comment in soup.findAll(text=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Function to check if a tag contains text
    def tag_contains_text(tag):
        if isinstance(tag, NavigableString):
            return tag.strip() != ""
        return any(tag_contains_text(child) for child in tag.children if not isinstance(child, Comment))

    # Remove tags that don't contain text or don't have children that contain text
    for tag in soup.find_all(True):
        if not tag_contains_text(tag):
            tag.decompose()
        else:
            # Strip all attributes from tags that contain text, expect hrefs on links
            href = tag.attrs.get("href")
            tag.attrs = {}

            if href is not None:
                tag.attrs["href"] = href

    return str(soup)


class PageQueryAnalysisResult(BaseModel):
    answer: str


class PageQueryAnalyzer(abc.ABC):
    @abc.abstractmethod
    def analyze(self, url: str, title: str, query: str, spinner: Optional[Halo] = None) -> PageQueryAnalysisResult:
        raise NotImplementedError()


class OpenAIChatPageQueryAnalyzer(PageQueryAnalyzer):
    def __init__(
        self,
        chat_model: BaseChatModel,
        page_retriever: PageRetriever,
        text_splitter: TextSplitter,
        use_first_split_only: bool = True,
    ):
        self.chat_model = chat_model
        self.page_retriever = page_retriever
        self.text_splitter = text_splitter
        self.use_first_split_only = use_first_split_only

    def analyze(self, url: str, title: str, query: str, spinner: Optional[Halo] = None) -> PageQueryAnalysisResult:
        try:
            html = self.page_retriever.retrieve_html(url)
        except (NonTransientHTTPError, TransientHTTPError) as e:
            return PageQueryAnalysisResult(
                answer=f"The query could not be answered because an error occurred while retrieving the page: {e}"
            )
        finally:
            self.page_retriever.close()

        cleaned_html = clean_html(html)

        docs = self.text_splitter.create_documents([cleaned_html])

        answer = "No answer yet."
        for i, doc in enumerate(docs):
            text = doc.page_content

            query_answerer = LangChainBasedAIChatParticipant(
                name="Web Page Query Answerer",
                role="Web Page Query Answerer",
                personal_mission="Answer queries based on provided (partial) web page content from the web.",
                chat_model=self.chat_model,
                other_prompt_sections=[
                    Section(
                        name="Crafting a Query Answer",
                        sub_sections=[
                            Section(
                                name="Process",
                                list=[
                                    "Analyze the query and the given content",
                                    "If context is provided, use it to answer the query.",
                                    "Summarize the answer in a comprehensive, yet succinct way.",
                                ],
                                list_item_prefix=None,
                            ),
                            Section(
                                name="Guidelines",
                                list=[
                                    "If the answer is not found in the page content, it's insufficent, or not relevant "
                                    "to the query at all, state it clearly.",
                                    "Do not fabricate information. Stick to provided content.",
                                    "Provide context for the next call (e.g., if a paragraph was cut short, include "
                                    "relevant header information, section, etc. for continuity). Assume the content is "
                                    "partial content from the page. Be very detailed in the context.",
                                    "If unable to answer but found important information, include it in the context "
                                    "for the next call.",
                                    "Pay attention to the details of the query and make sure the answer is suitable "
                                    "for the intent of the query.",
                                    "A potential answer might have been provided. This means you thought you found "
                                    "the answer in a previous partial text for the same page. You should double-check "
                                    "that and provide an alternative revised answer if you think it's wrong, "
                                    "or repeat it if you think it's right or cannot be validated using the current "
                                    "text.",
                                ],
                            ),
                        ],
                    )
                ],
            )

            final_answer, _ = get_response(
                query=str(
                    StructuredString(
                        sections=[
                            Section(name="Query", text=query),
                            Section(name="Url", text=url),
                            Section(name="Title", text=title),
                            Section(name="Previous Answer", text=answer),
                            Section(name="Page Content", text=f"```{text}```"),
                        ]
                    )
                ),
                answerer=query_answerer,
            )

            result = string_output_to_pydantic(
                output=final_answer, chat_model=self.chat_model, output_schema=PageQueryAnalysisResult
            )
            answer = result.answer

            if self.use_first_split_only:
                break

        return PageQueryAnalysisResult(
            answer=answer,
        )
