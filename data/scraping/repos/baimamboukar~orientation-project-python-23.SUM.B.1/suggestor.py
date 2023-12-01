from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


class Suggestor:
    """
    Summary
    ------
    suggestor static class for suggesting resume descriptions

    Attributes
    ----------
    chain (LLMChain): chat prompt chain

    Methods
    -------
    suggest(title: str, context: str, description: str) -> str
        suggest a description for a given title and context
    """

    system_prompt_template = SystemMessagePromptTemplate.from_template(
        "You are crafting a professional resume. You are working on a section about your time in {role}.\n"
        "These are the following activities that you have participated in during that time:\n\n{activities}"
    )

    human_prompt_template = HumanMessagePromptTemplate.from_template(
        "Please complete the description for this role:\n\n{description}"
    )

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_prompt_template, human_prompt_template]
    )

    chain = LLMChain(llm=ChatOpenAI(temperature=0.9), prompt=chat_prompt)

    @classmethod
    def suggest(cls, title: str, context: str, description: str) -> str:
        """
        Summary
        ------
        suggest a description for a given title and context

        Parameters
        ----------
        title (str): the title of the role
        context (str): the context of the role
        description (str): the description of the role

        Returns
        -------
        suggestion (str): the suggested description
        """
        return cls.chain.run(role=title, activities=context, description=description)
