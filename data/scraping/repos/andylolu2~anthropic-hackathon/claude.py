from dotenv import load_dotenv
from langchain.chat_models import ChatAnthropic
from langchain.prompts import PromptTemplate

from custom_parser import MarkdownOutputParser

load_dotenv()


class Claude:
    def __init__(self):
        self.model = ChatAnthropic(max_tokens=4096, temperature=0, cache=True)

    def ask_claude_md(self, query: str):
        parser = MarkdownOutputParser()
        prompt = PromptTemplate(
            template="""

            Human:
            {query}
            {format_instructions}
            Assistant:
            """,
            input_variables=["query"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | self.model | parser
        return chain.invoke({"query": query})
