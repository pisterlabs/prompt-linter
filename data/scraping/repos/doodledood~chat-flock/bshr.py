# Based directly on David Shaprio's BSHR Loop: https://github.com/daveshap/BSHR_Loop
from typing import Any, Dict, Generator, Generic, List, Optional, Type, TypeVar

import datetime
import json
from functools import partial

import questionary
from halo import Halo
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.chat_models.base import BaseChatModel
from langchain.llms.openai import OpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from chatflock.backing_stores import InMemoryChatDataBackingStore
from chatflock.backing_stores.langchain import LangChainMemoryBasedChatDataBackingStore
from chatflock.base import Chat, ChatDataBackingStore
from chatflock.conductors import RoundRobinChatConductor
from chatflock.parsing_utils import chat_messages_to_pydantic
from chatflock.participants.langchain import LangChainBasedAIChatParticipant
from chatflock.participants.user import UserChatParticipant
from chatflock.renderers import TerminalChatRenderer
from chatflock.sequencial_process import SequentialProcess, Step
from chatflock.structured_string import Section, StructuredString
from chatflock.use_cases.request_response import get_response
from chatflock.web_research import WebSearch
from chatflock.web_research.web_research import WebResearchTool


class BHSRState(BaseModel):
    information_need: Optional[str] = None
    queries_to_run: Optional[List[str]] = None
    answers_to_queries: Optional[Dict[str, str]] = None
    current_hypothesis: Optional[str] = None
    proposed_hypothesis: Optional[str] = None
    feedback: Optional[str] = None
    is_satisficed: Optional[bool] = None


def save_state(state: BHSRState, state_file: Optional[str]) -> None:
    if state_file is None:
        return

    data = state.model_dump()
    with open(state_file, "w") as f:
        json.dump(data, f, indent=2)


def load_state(state_file: Optional[str]) -> Optional[BHSRState]:
    if state_file is None:
        return None

    try:
        with open(state_file) as f:
            data = json.load(f)
            return BHSRState.model_validate(data)
    except FileNotFoundError:
        return None


class QueryGenerationResult(BaseModel):
    information_need: str = Field(description="Information need as requested by the user.")
    queries: List[str] = Field(description="Set of queries to run.")


class HypothesisGenerationResult(BaseModel):
    hypothesis: str = Field(
        description="A new or updated hypothesis based on the materials provided. Rich formatting using Markdown. Should include all relevant citations inline."
    )


class SatisficationCheckResult(BaseModel):
    feedback: str = Field(
        description="If not satisficed yet, feedback on why not satisfied and what to think about next. If satisficed, feedback can be empty."
    )
    is_satisficed: bool = Field(description="Whether or not the information need has been satisficed.")


def generate_queries(
    state: BHSRState,
    chat_model: BaseChatModel,
    interactive_user: bool = True,
    max_queries: int = 5,
    shared_sections: Optional[List[Section]] = None,
    web_search_tool: Optional[BaseTool] = None,
    spinner: Optional[Halo] = None,
) -> None:
    if state.queries_to_run is not None and len(state.queries_to_run) > 0:
        # Means we are continuing a previous session
        return

    if shared_sections is None:
        shared_sections = []

    query_generator = LangChainBasedAIChatParticipant(
        name="Search Query Generator",
        role="Search Query Generator",
        personal_mission="You will be given a specific query or problem by the user and you are to generate a list of "
        f"AT MOST {max_queries} search queries that will be used to search the internet. Make sure you "
        f"generate comprehensive, counterfactual, and maximally orthogonal search queries. "
        "Employ everything you know about "
        "information foraging and information literacy to generate the best possible questions. "
        "Use a step-by-step approach and think about the information need and the information "
        "domain before generating the queries. Order the queries by their importance and relevance "
        "to the main information need of the user.",
        other_prompt_sections=shared_sections
        + [
            Section(
                name="Unclear Information Need",
                text=(
                    "If the information need or query are vague and unclear, either perform a web search to "
                    "clarify the information need or ask the user for clarification."
                    if interactive_user
                    else "If the information need or query are vague and unclear, either perform a web search to "
                    "clarify the information need or make a best guess. The user will not be available to "
                    "respond back."
                ),
            ),
            Section(
                name="Refine Queries",
                text='You might be given a first-pass information need with "None" previous queries and answers, '
                "in which case you will do the best you"
                'can to generate "naive queries" (uninformed search queries). However the USER might also '
                "give you previous search queries or other background information such as accumulated notes. "
                'If these materials are present, you are to generate "informed queries" - more specific '
                "search queries that aim to zero in on the correct information domain. Do not duplicate "
                "previously asked questions. Use the notes and other information presented to create "
                "targeted queries and/or to cast a wider net.",
            ),
            Section(
                name="Termination",
                text="Once you generate a new set of queries to run, you should terminate the chat immediately by "
                "ending your message with TERMINATE",
            ),
        ],
        tools=[web_search_tool] if web_search_tool is not None else None,
        ignore_group_chat_environment=True,
        chat_model=chat_model,
        spinner=spinner,
    )

    user = UserChatParticipant()
    participants = [user, query_generator]

    try:
        memory = ConversationSummaryBufferMemory(
            llm=chat_model, max_token_limit=OpenAI.modelname_to_contextsize(chat_model.model_name)  # type: ignore
        )
        backing_store: ChatDataBackingStore = LangChainMemoryBasedChatDataBackingStore(memory=memory)
    except ValueError:
        backing_store = InMemoryChatDataBackingStore()

    chat = Chat(
        backing_store=backing_store,
        renderer=TerminalChatRenderer(),
        initial_participants=participants,
        max_total_messages=None if interactive_user else 2,
    )

    chat_conductor = RoundRobinChatConductor()

    if state.information_need is None:
        if spinner is not None:
            spinner.stop()

        _ = chat_conductor.initiate_dialog(
            chat=chat, initial_message=f"What is your information need or query?", from_participant=query_generator
        )
    else:
        _ = chat_conductor.initiate_dialog(
            chat=chat,
            initial_message=str(
                StructuredString(
                    sections=[
                        Section(name="Information Need", text=state.information_need),
                        Section(
                            name="Previous Queries & Answers",
                            text="None"
                            if state.answers_to_queries is None or len(state.answers_to_queries) == 0
                            else None,
                            sub_sections=[
                                Section(name=query, text=f"```markdown\n{answer}\n```", uppercase_name=False)
                                for query, answer in (state.answers_to_queries or {}).items()
                            ],
                        ),
                        Section(name="Current Hypothesis", text=str(state.current_hypothesis)),
                    ]
                )
            ),
            from_participant=user,
        )

    output = chat_messages_to_pydantic(
        chat_messages=chat.get_messages(), chat_model=chat_model, output_schema=QueryGenerationResult
    )

    if state.information_need is None:
        state.information_need = output.information_need

    if state.queries_to_run is None:
        state.queries_to_run = []

    state.queries_to_run += output.queries[:max_queries]


def search_queries(
    state: BHSRState, web_search: WebSearch, n_search_results: int = 3, spinner: Optional[Halo] = None
) -> Generator[BHSRState, None, None]:
    if state.queries_to_run is None:
        return

    queries_and_answers = state.answers_to_queries if state.answers_to_queries is not None else {}
    queries_to_run_set = set(state.queries_to_run)
    for query in state.queries_to_run:
        if query in queries_and_answers:
            continue

        answer = web_search.get_answer(query=query, n_results=n_search_results, spinner=spinner)[1]

        queries_and_answers[query] = answer
        queries_to_run_set.remove(query)

        state.answers_to_queries = queries_and_answers
        state.queries_to_run = list(queries_to_run_set)

        yield state


def generate_hypothesis(
    state: BHSRState,
    chat_model: BaseChatModel,
    shared_sections: Optional[List[Section]] = None,
    spinner: Optional[Halo] = None,
) -> None:
    hypothesis_generator = LangChainBasedAIChatParticipant(
        name="Information Needs Hypothesis Generator",
        role="Information Needs Hypothesis Generator",
        personal_mission="You are an information needs hypothesis generator. You will be given a main information "
        "need or user query as well as a variety of materials, such as search results, "
        "previous hypotheses, and notes. Whatever information you receive, your output should be a "
        "revised, refined, or improved hypothesis. In this case, the hypothesis is a comprehensive "
        "answer to the user query or information need. To the best of your ability. Do not include "
        "citations in your hypothesis, as this will all be record via out-of-band processes (e.g. "
        "the information that you are shown will have metadata and cataloging working behind the "
        "scenes that you do not see). Even so, you should endeavour to write everything in complete, "
        "comprehensive sentences and paragraphs such that your hypothesis requires little to no "
        "outside context to understand. Your hypothesis must be relevant to the USER QUERY or "
        "INFORMATION NEED. Format the hypothesis in rich markdown and include all relevant citations "
        "inline.",
        other_prompt_sections=shared_sections,
        ignore_group_chat_environment=True,
        chat_model=chat_model,
        spinner=spinner,
    )

    _, chat = get_response(
        query=str(
            StructuredString(
                sections=[
                    Section(name="Information Need", text=state.information_need),
                    Section(
                        name="Previous Queries & Answers",
                        text="None" if state.answers_to_queries is None or len(state.answers_to_queries) == 0 else None,
                        sub_sections=[
                            Section(name=query, text=f"```markdown\n{answer}\n```", uppercase_name=False)
                            for query, answer in (state.answers_to_queries or {}).items()
                        ],
                    ),
                    Section(name="Previous Hypothesis", text=str(state.current_hypothesis)),
                    Section(name="Feedback", text=str(state.feedback)),
                ]
            )
        ),
        answerer=hypothesis_generator,
        renderer=TerminalChatRenderer(),
    )
    output = chat_messages_to_pydantic(
        chat_messages=chat.get_messages(), chat_model=chat_model, output_schema=HypothesisGenerationResult
    )

    state.proposed_hypothesis = output.hypothesis


def check_satisficing(
    state: BHSRState,
    chat_model: BaseChatModel,
    shared_sections: Optional[List[Section]] = None,
    spinner: Optional[Halo] = None,
) -> None:
    satisficing_checker = LangChainBasedAIChatParticipant(
        name="Information Needs Satisficing Checker",
        role="Information Needs Satisficing Checker",
        personal_mission="You are an information needs satisficing checker. You will be given a litany of materials, "
        "including an original user query, previous search queries, their results, notes, "
        "and a final hypothesis. You are to generate a decision as to whether or not the information "
        "need has been satisficed or not. You are to make this judgment by virtue of several "
        "factors: amount and quality of searches performed, specificity and comprehensiveness of the "
        "hypothesis, and notes about the information domain and foraging (if present). Several "
        "things to keep in mind: the user's information need may not be answerable, "
        "or only partially answerable, given the available information or nature of the problem.  "
        "Unanswerable data needs are satisficed when data foraging doesn't turn up more relevant "
        "information. Use a step-by-step approach to determine whether or not the information need "
        "has been satisficed.",
        other_prompt_sections=shared_sections,
        ignore_group_chat_environment=True,
        chat_model=chat_model,
        spinner=spinner,
    )

    _, chat = get_response(
        query=str(
            StructuredString(
                sections=[
                    Section(name="Information Need", text=state.information_need),
                    Section(
                        name="Previous Queries & Answers",
                        text="None" if state.answers_to_queries is None or len(state.answers_to_queries) == 0 else None,
                        sub_sections=[
                            Section(name=query, text=f"```markdown\n{answer}\n```", uppercase_name=False)
                            for query, answer in (state.answers_to_queries or {}).items()
                        ],
                    ),
                    Section(name="Previous Hypothesis", text=str(state.current_hypothesis)),
                    Section(name="Proposed New Hypothesis", text=str(state.proposed_hypothesis)),
                ]
            )
        ),
        answerer=satisficing_checker,
    )
    output = chat_messages_to_pydantic(
        chat_messages=chat.get_messages(), chat_model=chat_model, output_schema=SatisficationCheckResult
    )

    state.feedback = output.feedback
    state.is_satisficed = output.is_satisficed
    state.current_hypothesis = state.proposed_hypothesis
    state.proposed_hypothesis = None


def brainstorm_search_hypothesize_refine(
    web_search: WebSearch,
    chat_model: BaseChatModel,
    initial_state: Optional[BHSRState] = None,
    n_search_results: int = 3,
    state_file: Optional[str] = None,
    spinner: Optional[Halo] = None,
) -> BHSRState:
    shared_sections = [Section(name="Current Date (YYYY-MM-DD)", text=datetime.datetime.utcnow().strftime("%Y-%m-%d"))]
    web_search_tool = WebResearchTool(web_search=web_search, n_results=n_search_results, spinner=spinner)

    if state_file is not None and spinner is not None:
        spinner.start("Loading previous state...")

    initial_state = BHSRState() if initial_state is None else initial_state

    process = SequentialProcess(
        steps=[
            Step(
                name="Query Generation",
                func=partial(
                    generate_queries,
                    chat_model=chat_model,
                    interactive_user=initial_state.information_need is None,
                    shared_sections=shared_sections,
                    web_search_tool=web_search_tool,
                    spinner=spinner,
                ),
                on_step_start=lambda _: spinner.start("Generating queries...") if spinner is not None else None,
                on_step_completed=lambda _: spinner.succeed("Queries generated.") if spinner is not None else None,
            ),
            Step(
                name="Web Search",
                func=partial(search_queries, web_search=web_search, n_search_results=n_search_results, spinner=spinner),
                on_step_start=lambda _: spinner.start("Searching queries...") if spinner is not None else None,
                on_step_completed=lambda _: spinner.succeed("Queries answered.") if spinner is not None else None,
            ),
            Step(
                name="Hypothesis Generation",
                func=partial(
                    generate_hypothesis, chat_model=chat_model, shared_sections=shared_sections, spinner=spinner
                ),
                on_step_start=lambda _: spinner.start("Generating hypothesis...") if spinner is not None else None,
                on_step_completed=lambda _: spinner.succeed("Hypothesis generated.") if spinner is not None else None,
            ),
            Step(
                name="Satificing Check",
                func=partial(
                    check_satisficing, chat_model=chat_model, shared_sections=shared_sections, spinner=spinner
                ),
                on_step_start=lambda _: spinner.start("Checking satisfication condition...")
                if spinner is not None
                else None,
                on_step_completed=lambda _: spinner.succeed("Satisfication checked.") if spinner is not None else None,
            ),
        ],
        initial_state=initial_state,
        save_state=partial(save_state, state_file=state_file),
    )

    state = process.run()
    return state  # type: ignore


def run_brainstorm_search_hypothesize_refine_loop(
    web_search: WebSearch,
    chat_model: BaseChatModel,
    n_search_results: int = 3,
    initial_state: Optional[BHSRState] = None,
    state_file: Optional[str] = None,
    confirm_satisficed: bool = False,
    spinner: Optional[Halo] = None,
) -> str:
    loaded_state = load_state(state_file)
    if loaded_state is None:
        initial_state = BHSRState() if initial_state is None else initial_state

        if spinner is not None:
            spinner.stop()
    else:
        initial_state = loaded_state
        if spinner is not None:
            spinner.succeed("Loaded previous state.")

        if initial_state.is_satisficed:
            spinner.warn("The information need has already been satisficed")

            return initial_state.current_hypothesis or ""

    while True:
        state = brainstorm_search_hypothesize_refine(
            initial_state=initial_state,
            web_search=web_search,
            chat_model=chat_model,
            n_search_results=n_search_results,
            state_file=state_file,
            spinner=spinner,
        )

        if state.is_satisficed:
            if not confirm_satisficed:
                break

            has_feedback = questionary.confirm(
                "The information need seems to have been satisficed. Do you have any feedback?"
            ).ask()

            if not has_feedback:
                break

            feedback = questionary.text("What is your feedback?").ask()

            state.is_satisficed = False
            state.feedback = feedback

    return state.current_hypothesis or ""


class BrainstormSearchHypothesizeRefineToolArgs(BaseModel):
    query: str = Field(description="The query to thoroughly research.")


TArgSchema = TypeVar("TArgSchema", bound=BaseModel)


class BrainstormSearchHypothesizeRefineTool(BaseTool, Generic[TArgSchema]):
    web_search: WebSearch
    chat_model: BaseChatModel
    n_results: int = 3
    state_file: Optional[str] = None
    spinner: Optional[Halo] = None
    name: str = "web_research"
    description: str = (
        "Research the web using a Brainstorm-Search-Hypothesize-Refine approach. Use that to get a "
        "very comprehensive (but expensive) answer for a query you don't know or unsure of the answer "
        "to, for recent events, or if the user asks you to. This will evaluate answer snippets, "
        "knowledge graphs, and the top N results from google and aggregate a result for multiple "
        "queries. Very thorough research."
    )
    args_schema: Type[TArgSchema] = BrainstormSearchHypothesizeRefineToolArgs  # type: ignore
    progress_text: str = "Researching the topic (this may take a while)..."

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None, **kwargs: Any) -> Any:
        hypothesis = run_brainstorm_search_hypothesize_refine_loop(
            initial_state=BHSRState(information_need=query),
            confirm_satisficed=False,
            web_search=self.web_search,
            chat_model=self.chat_model,
            n_search_results=self.n_results,
            state_file=self.state_file,
            spinner=self.spinner,
        )

        return hypothesis
