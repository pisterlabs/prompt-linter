# Based directly on David Shaprio's BSHR Loop: https://github.com/daveshap/BSHR_Loop
from typing import Optional

from pathlib import Path

import typer
from dotenv import load_dotenv
from halo import Halo
from langchain.text_splitter import TokenTextSplitter

from chatflock.use_cases.bshr import run_brainstorm_search_hypothesize_refine_loop
from chatflock.web_research import OpenAIChatPageQueryAnalyzer, WebSearch
from chatflock.web_research.page_retrievers import SeleniumPageRetriever
from chatflock.web_research.search import GoogleSerperSearchResultsProvider
from examples.common import create_chat_model, get_max_context_size


def bshr_loop(
    model: str = "gpt-4-1106-preview",
    model_for_page_analysis: str = "gpt-3.5-turbo-1106",
    temperature: float = 0.0,
    temperature_for_page_analysis: float = 0.0,
    n_search_results: int = 3,
    state_file_path: Optional[str] = "output/bshr_state.json",
) -> None:
    if state_file_path is not None:
        Path(state_file_path).parent.mkdir(exist_ok=True, parents=True)

    chat_model = create_chat_model(model=model, temperature=temperature)
    chat_model_for_analysis = create_chat_model(
        model=model_for_page_analysis, temperature=temperature_for_page_analysis
    )

    max_context_size = get_max_context_size(chat_model_for_analysis) or 12_000

    page_retriever = SeleniumPageRetriever()
    web_search = WebSearch(
        chat_model=chat_model,
        # Make sure you have a valid API Key for Serper in your .env file: SERPER_API_KEY=...
        search_results_provider=GoogleSerperSearchResultsProvider(),
        page_query_analyzer=OpenAIChatPageQueryAnalyzer(
            chat_model=chat_model_for_analysis,
            page_retriever=page_retriever,
            text_splitter=TokenTextSplitter(chunk_size=max_context_size, chunk_overlap=max_context_size // 5),
            use_first_split_only=True,
        ),
    )

    spinner = Halo(spinner="dots")

    hypothesis = run_brainstorm_search_hypothesize_refine_loop(
        confirm_satisficed=True,
        web_search=web_search,
        chat_model=chat_model,
        n_search_results=n_search_results,
        state_file=state_file_path,
        spinner=spinner,
    )

    print(f"Final Answer:\n----------------\n{hypothesis}\n----------------")

    page_retriever.close()


if __name__ == "__main__":
    load_dotenv()

    typer.run(bshr_loop)
