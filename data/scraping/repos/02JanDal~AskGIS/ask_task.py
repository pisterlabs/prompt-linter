from dataclasses import dataclass
from typing import Optional

from langchain import OpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import BaseCallbackHandler, CallbackManager
from PyQt5.QtCore import pyqtSignal
from qgis.core import QgsProcessingFeedback, QgsTask

from askgis.lib.chain import GISTool
from askgis.lib.context import Context


@dataclass
class AskResult:
    answer: str


class AskTask(QgsTask):
    codeChanged = pyqtSignal(str)
    promptChanged = pyqtSignal(str)

    def __init__(
        self,
        description: str,
        question: str,
        context: Context,
        api_key: str,
        personality: str,
        callbacks: BaseCallbackHandler,
    ) -> None:
        super().__init__(description)
        self._question = question
        self._context = context
        self._api_key = api_key
        self._personality = personality
        self._callbacks = callbacks
        self._exception: Optional[Exception] = None
        self._result: Optional[AskResult] = None

    @property
    def result(self) -> Optional[AskResult]:
        return self._result

    def run(self) -> bool:
        feedback = QgsProcessingFeedback()
        feedback.progressChanged.connect(self.setProgress)

        try:
            llm = OpenAI(temperature=0, openai_api_key=self._api_key)

            callback_manager = CallbackManager([self._callbacks])

            gis_tool = GISTool(
                context=self._context,
                llm=llm,
                feedback=feedback,
                callback_manager=callback_manager,
                code_callback=self.codeChanged.emit,
                prompt_callback=self.promptChanged.emit,
            )

            tools = load_tools(["llm-math"], llm=llm, callback_manager=callback_manager)
            agent = initialize_agent(
                [gis_tool, *tools],
                llm,
                AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                callback_manager=callback_manager,
            )
            answer = agent.run(self._question)
            self._result = AskResult(answer=answer)
            return True
        except Exception as e:
            self._exception = e
            return False

    def finished(self, result: bool) -> None:
        if not result and self._exception:
            raise self._exception
