import logging
from langchain.base_language import BaseLanguageModel
from langchain.chains.base import Chain
from langchain.callbacks.manager import CallbackManagerForChainRun
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from ..paper import Paper, search_on_google_scholar, create_papers_vectorstor
from .outline import SROutlintChain, Outlint, Section
from .overview import SROverviewChain, Overview
from .section import SRSectionChain

logger = logging.getLogger(__name__)


class SRChain(Chain):

    llm: BaseLanguageModel
    output_key: str = "text"

    @property
    def input_keys(self) -> List[str]:
        return ["query"]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        query = inputs["query"]
        logger.info(f"Searching `{query}` on Google Scholar.")
        papers = search_on_google_scholar(query)

        logger.info(f"Writing an overview of the paper.")
        overview_chain = SROverviewChain(llm=self.llm, verbose=self.verbose)
        overview: Overview = overview_chain.run({"query": query, "papers": papers})

        logger.info(f"Building the outline of the paper.")
        outline_chain = SROutlintChain(llm=self.llm, verbose=self.verbose)
        outline: Outlint = outline_chain.run({
            "query": query,
            "papers": papers,
            "overview": overview
        })

        logger.info(f"Creating vector store.")
        db = create_papers_vectorstor(papers)

        section_chain = SRSectionChain(llm=self.llm, paper_store=db, verbose=self.verbose)
        flatten_sections = get_flatten_sections(outline)
        sections_as_md = []

        for section_idx in range(len(flatten_sections)):
            logger.info(f"Writing sections: [{section_idx + 1} / {len(flatten_sections)}]")

            sections_as_md.append(
                section_chain.run({
                    "section_idx": section_idx,
                    "query": query,
                    "papers": papers,
                    "overview": overview,
                    "outline": outline,
                    "flatten_sections": flatten_sections,
                })
            )

        return {
            self.output_key: create_output(outline, overview, papers, flatten_sections, sections_as_md)
        }


class FlattenSection(BaseModel):

    """SRChain 向けのセクションを表すヘルパークラス
    """

    level: int
    section: Section


def get_flatten_sections(
        outline: Outlint,
        start_section_level: int = 2,
) -> List[FlattenSection]:
    def inner(section_level, section: Section) -> List[FlattenSection]:
        result = FlattenSection(level=section_level, section=section)

        if not section.children:
            return [result]

        return (
            [result] + sum([
                inner(section_level + 1, child)
                for child in section.children
            ], [])
        )

    return sum([
        inner(start_section_level, section)
        for section in outline.sections
    ], [])


def create_output(
        outline: Outlint,
        overview: Overview,
        papers: List[Paper],
        flatten_sections: List[FlattenSection],
        sections_as_md: List[str],
) -> str:
    papers_citation_id_map = {p.citation_id: p for p in papers}
    all_citation_ids = list(set(
        outline.citations_ids + sum([
            s.section.citation_ids for s in flatten_sections
        ], [])
    ))

    citations = []

    for citation_id in all_citation_ids:
        citation = papers_citation_id_map[int(citation_id)]
        citations.append(
            f"[^{citation_id}]: "
            f"[{citation.mla_citiation.snippet}]({citation.link})"
        )

    return (
        f"# {overview.title}\n\n{overview.overview}\n\n"
        + f"## Table of contents\n\n{outline}\n\n"
        + "\n\n".join(sections_as_md)
        + "\n\n## References\n"
        + "\n\n".join(citations)
    )
