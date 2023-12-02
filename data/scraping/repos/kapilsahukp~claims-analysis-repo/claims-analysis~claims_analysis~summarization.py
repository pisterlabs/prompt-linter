from dataclasses import dataclass

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from claims_analysis.constants import SUMMARIZATION_PROMPT
from claims_analysis.page_processing import Violation


@dataclass
class ClaimSummary:
    """For storing a summary of findings for a claim."""

    filepath: str
    pages_total: int
    pages_processed: int
    pages_flagged: int
    summary: str


def summarize_results(violations: list[Violation], temperature: float = 0) -> str:
    """Given page level results, create a summary of the potential reasons for policy violation."""

    # Extract only the relevant parts of the violations
    simplified_violations = [
        "(page_no={}, issue_desc='{}')".format(violation.page_no, violation.issue_desc)
        for violation in violations
    ]
    violations_str = "Potential violations: [" + ", ".join(simplified_violations) + "]"

    # Send to API for summary
    chat = ChatOpenAI(temperature=temperature, model_name="gpt-3.5-turbo", client=None)
    messages = [
        SystemMessage(content=SUMMARIZATION_PROMPT),
        HumanMessage(content=violations_str),
    ]

    return chat(messages).content
