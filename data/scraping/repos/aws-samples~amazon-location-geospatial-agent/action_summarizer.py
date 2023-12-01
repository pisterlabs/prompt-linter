import json
from typing import List, Any, Optional

from langchain import PromptTemplate, LLMChain
from pydantic import BaseModel, ConfigDict
from pydispatch import dispatcher

from geospatial_agent.agent.action_summarizer.prompts import _ACTION_SUMMARY_PROMPT, _ROLE_INTRO, \
    _READ_FILE_PROMPT, _READ_FILE_REQUIREMENTS, _ACTION_SUMMARY_REQUIREMENTS, DATA_FRAMES_VARIABLE_NAME, \
    _DATA_SUMMARY_REQUIREMENTS, _DATA_SUMMARY_PROMPT
from geospatial_agent.agent.shared import AgentSignal, EventType, SIGNAL_ACTION_CONTEXT_GENERATED, \
    SENDER_ACTION_SUMMARIZER, SIGNAL_FILE_READ_CODE_GENERATED, SIGNAL_FILE_READ_CODE_EXECUTED
from geospatial_agent.shared.bedrock import get_claude_v2
from geospatial_agent.shared.prompts import HUMAN_ROLE, ASSISTANT_ROLE, HUMAN_STOP_SEQUENCE
from geospatial_agent.shared.shim import get_shim_imports
from geospatial_agent.shared.utils import extract_code


class ActionSummarizerException(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class ActionContext(BaseModel):
    action: str
    file_paths: List[str]


class FileSummary(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    file_url: str
    data_frame: Any
    column_names: List[str]
    file_summary: Optional[str] = None


class ActionSummary(BaseModel):
    action: str
    file_summaries: List[FileSummary]


class ActionSummarizer:
    """Action summarizer acts on raw user messages with the following traits
    1. It is a geospatial query or analysis such as "Draw me a heatmap".
    2. Has URLS of data to be used for the analysis.

    ActionSummarizer generates a list of ActionSummary.
    """

    def __init__(self, llm=None):
        if llm is None:
            claude_v2 = get_claude_v2()
            self.llm = claude_v2
        else:
            self.llm = llm

    def invoke(self, user_input: str, session_id: str, storage_mode: str) -> ActionSummary:
        try:
            action_context = self._extract_action_context(user_input)
            dispatcher.send(signal=SIGNAL_ACTION_CONTEXT_GENERATED,
                            sender=SENDER_ACTION_SUMMARIZER,
                            event_data=AgentSignal(
                                event_type=EventType.Message,
                                event_source=SENDER_ACTION_SUMMARIZER,
                                event_message=f'Detected desired action {action_context.action}. And file paths: {action_context.file_paths}.'
                            ))

            read_file_code = self._gen_file_read_code(action_context, session_id, storage_mode)
            dispatcher.send(signal=SIGNAL_FILE_READ_CODE_GENERATED,
                            sender=SENDER_ACTION_SUMMARIZER,
                            event_data=AgentSignal(
                                event_type=EventType.PythonCode,
                                event_source=SENDER_ACTION_SUMMARIZER,
                                event_message=f'Generated code to read and understand data schema.',
                                event_data=read_file_code
                            ))

            data_files_summary = self._gen_file_summaries_from_executing_code(read_file_code)
            dispatcher.send(signal=SIGNAL_FILE_READ_CODE_EXECUTED,
                            sender=SENDER_ACTION_SUMMARIZER,
                            event_data=AgentSignal(
                                event_type=EventType.Message,
                                event_source=SENDER_ACTION_SUMMARIZER,
                                event_message=f'Successfully executed code to read and understand data schema.',
                            ))

            file_summaries = self._gen_file_summaries_for_action(action_context.action, data_files_summary)
            return ActionSummary(action=action_context.action, file_summaries=file_summaries)

        except Exception as e:
            if isinstance(e, ActionSummarizerException):
                raise e
            else:
                raise ActionSummarizerException(
                    message=f"Failed to extract dataframes from data reading code. Original exception: {e}") from e

    def _gen_file_summaries_for_action(self, action: str, file_summaries: List[FileSummary]) -> List[FileSummary]:
        for item in file_summaries:
            requirements_str = "\n".join(
                [f"{index + 1}. {requirement}" for index, requirement in enumerate(_DATA_SUMMARY_REQUIREMENTS)])
            file_summary_template: PromptTemplate = PromptTemplate.from_template(_DATA_SUMMARY_PROMPT)
            gdf_str = item.data_frame.to_json()

            if len(gdf_str) > 4000:
                gdf_str = gdf_str[:4000]

            chain = LLMChain(llm=self.llm, prompt=file_summary_template)
            file_summary = chain.run(
                role_intro=_ROLE_INTRO,
                human_role=HUMAN_ROLE,
                requirements=requirements_str,
                action=action,
                columns=item.column_names,
                table=gdf_str,
                assistant_role=ASSISTANT_ROLE,
                stop=[HUMAN_STOP_SEQUENCE]
            ).strip()
            item.file_summary = file_summary

        return file_summaries

    def _gen_file_read_code(self, action_context: ActionContext, session_id: str, storage_mode: str) -> str:
        file_paths = action_context.file_paths
        file_urls_str = "\n".join(
            [f"{index + 1}. {file_url}" for index, file_url in enumerate(file_paths)])

        requirements_str = "\n".join(
            [f"{index + 1}. {requirement}" for index, requirement in enumerate(_READ_FILE_REQUIREMENTS)])
        read_file_template: PromptTemplate = PromptTemplate.from_template(_READ_FILE_PROMPT)

        chain = LLMChain(llm=self.llm, prompt=read_file_template)
        read_file_code_response = chain.run(
            role_intro=_ROLE_INTRO,
            human_role=HUMAN_ROLE,
            requirements=requirements_str,
            session_id=session_id,
            storage_mode=storage_mode,
            assistant_role=ASSISTANT_ROLE,
            file_urls=file_urls_str,
            stop=[HUMAN_STOP_SEQUENCE]
        ).strip()

        read_file_code = extract_code(read_file_code_response)
        return read_file_code

    @staticmethod
    def _gen_file_summaries_from_executing_code(code: str) -> List[FileSummary]:
        assembled_code = f'{get_shim_imports()}\n{code}'
        output = exec(assembled_code, globals(), globals())
        _globals = globals()

        dataframes = _globals[DATA_FRAMES_VARIABLE_NAME]
        file_summaries = [FileSummary(**data) for data in dataframes]

        if len(file_summaries) == 0:
            raise ActionSummarizerException(
                message=f"Failed to generate file summaries from executing code. "
                        f"No dataframes found in globals")

        for item in file_summaries:
            if not isinstance(item.file_url, str):
                raise ActionSummarizerException(
                    message=f"Failed to generate file summaries from executing code. "
                            f"Found {type(item.file_url)} instead of str")
            if not isinstance(item.column_names, list):
                raise ActionSummarizerException(
                    message=f"Failed to generate file summaries from executing code. "
                            f"Found {type(item.column_names)} instead of list")

        return file_summaries

    def _extract_action_context(self, user_input: str) -> ActionContext:
        filepaths_extract_template: PromptTemplate = PromptTemplate.from_template(_ACTION_SUMMARY_PROMPT)
        requirements_str = "\n".join(
            [f"{index + 1}. {requirement}" for index, requirement in enumerate(_ACTION_SUMMARY_REQUIREMENTS)])

        chain = LLMChain(llm=self.llm, prompt=filepaths_extract_template)
        action_summary = chain.run(
            role_intro=_ROLE_INTRO,
            human_role=HUMAN_ROLE,
            requirements=requirements_str,
            assistant_role=ASSISTANT_ROLE,
            message=user_input,
            stop=[HUMAN_STOP_SEQUENCE]
        ).strip()

        try:
            action_summary_obj = ActionContext.parse_raw(action_summary)
            return action_summary_obj
        except json.JSONDecodeError as e:
            raise ValueError("Invalid JSON format.") from e
