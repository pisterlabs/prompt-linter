import json

from council.contexts import SkillContext, ChatMessage
from council.skills import SkillBase
from council.skills.google import GoogleSearchSkill, GoogleNewsSkill

from pandasai import PandasAI
from pandasai.llm.openai import OpenAI

import constants
from retrieval import Retriever
from utils import get_filename, read_file_to_df


class DocRetrievalSkill(SkillBase):
    """Skill to retrieve documents and build context"""

    def __init__(self, retriever: Retriever):
        super().__init__(name="document_retrieval")
        self.retriever = retriever

    def execute(self, context: SkillContext) -> ChatMessage:
        query = context.current.last_message.message
        context = self.retriever.retrieve_docs(query)
        return self.build_success_message(context)


class CustomGoogleSearchSkill(GoogleSearchSkill):
    """
    A skill that performs a Google search using the reformulated query from the controller.

    Based on GoogleSearchSkill: https://github.com/chain-ml/council/blob/main/council/skills/google/google_search_skill.py
    """

    def execute(self, context: SkillContext) -> ChatMessage:
        # Execute the skill only if the API keys required for Google Search are provided
        if self.gs:
            prompt = context.current.last_message
            resp = self.gs.execute(query=prompt.message, nb_results=5)
            response_count = len(resp)
            if response_count > 0:
                return self.build_success_message(
                    f"{self._name} {response_count} responses for {prompt.message}",
                    json.dumps([r.dict() for r in resp]),
                )
            return self.build_error_message("no response")
        return self.build_error_message("API keys for Google Search not provided")


class CustomGoogleNewsSkill(GoogleNewsSkill):
    """
    A skill that performs a Google News search using the reformulated query from the controller.

    Based on GoogleNewsSkill: https://github.com/chain-ml/council/blob/main/council/skills/google/google_news_skill.py
    """

    def execute(self, context: SkillContext) -> ChatMessage:
        prompt = context.current.last_message
        resp = self.gn.execute(query=prompt.message, nb_results=5)
        response_count = len(resp)
        if response_count > 0:
            return self.build_success_message(
                f"gnews {response_count} responses for {prompt.message}",
                json.dumps([r.dict() for r in resp]),
            )
        return self.build_error_message("no response")


class GoogleAggregatorSkill(SkillBase):
    """Skill to aggregate results from Google Search and Google News"""

    def __init__(
        self,
    ):
        super().__init__(name="google_aggregator")

    def execute(self, context: SkillContext) -> ChatMessage:
        gsearch_results = (
            json.loads(context.current.last_message_from_skill("gsearch").data)
            if context.current.last_message_from_skill("gsearch").is_ok
            else []
        )
        gnews_results = (
            json.loads(context.current.last_message_from_skill("gnews").data)
            if context.current.last_message_from_skill("gnews").is_ok
            else []
        )
        search_results = gsearch_results + gnews_results

        context = ""
        for result in search_results:
            text = result.get("title", "") + " " + result.get("snippet", "") + "\n\n"
            context += text

        return self.build_success_message(context)


class PandasSkill(SkillBase):
    """
    Skill to converse with pandas Dataframe using PandasAI.

    PandasAI: https://github.com/gventuri/pandas-ai/tree/main
    """

    def __init__(self, api_token, model):
        super().__init__(name="pandas")
        self.llm = OpenAI(api_token=api_token, model=model)

    def execute(self, context: SkillContext) -> ChatMessage:
        query = context.current.last_message.message

        df = read_file_to_df(get_filename(constants.MARKET_DATA_DIR))
        pandas_ai = PandasAI(self.llm, conversational=True)

        try:
            response = pandas_ai(data_frame=df, prompt=query)
            if (
                "Unfortunately, I was not able to answer your question, because of the following error:"
                in response
            ):
                return self.build_error_message(response)
            return self.build_success_message(response)
        except Exception as e:
            return self.build_error_message(
                f"PandasAI failed due to following error: {e}"
            )
