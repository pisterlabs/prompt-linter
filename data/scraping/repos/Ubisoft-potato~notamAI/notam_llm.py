from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    AIMessage,
    SystemMessage,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)


class NOTAMLLMChat:
    librarian_prompt = SystemMessage(
        content="You are a NOTAM Librarian. I will give you a series of NOTAM messages and a list of NOTAM Tags."
                "For each series of NOTAM messages, create a markdown format table with the columns below. "
                "Each NOTAM should be on one row. Columns:"
                "A. **NOTAM** - NOTAM ICAO code and ID. Add one asterisk (*) before and after the NOTAM."
                "B. **Explained** - In very simple English only, explain the NOTAM in 4 words or less."
                "Do not use abbreviations. Use sentence case."
                "C. **Tag**. Choose the most logical Tag for this NOTAM from the list of Tags."
                "Format as Tag Code - Tag Name. Add two asterisks (**) before and after the Tag.")

    notam_tags_message = HumanMessagePromptTemplate.from_template("List of NOTAM Tags,in three columns:\n"
                                                                  "Tag Code  Tag Name  Tag Description\n"
                                                                  "{tags}\n"
                                                                  "Read and wait, no action yet.")
    ai_wait_resp = AIMessage(content="Understood. Waiting for the NOTAM messages.")
    notam_messages = HumanMessagePromptTemplate.from_template("{notams}")

    def __init__(self, verbose: bool = False):
        self.chat = ChatOpenAI()
        self.llm_chain = LLMChain(
            llm=self.chat,
            prompt=ChatPromptTemplate.from_messages([self.librarian_prompt, self.notam_tags_message, self.ai_wait_resp,
                                                     self.notam_messages]),
            verbose=verbose
        )

    def chat_to_get_notam_about(self, notam_tags, notam_messages: str) -> str:
        return self.llm_chain.run({
            "tags": notam_tags,
            "notams": notam_messages
        })
