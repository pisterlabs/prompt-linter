import asyncio

# import llm
import textwrap
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI


class Adventure:
    system_def = "You are a talented author of Choose-Your-Own-Adventure books"
    temperature = 0.7
    end_of_first_part = "And now, turn the page."
    general_formatting_style = """
Write several paragraphs separated by blank lines.

Remarque: 
- Do not mention the page numbers! 
- Do Not give any choices to the reader yet.
"""

    def __init__(self, api_key):
        # self.model = llm.get_model("gpt-3.5-turbo-16k")
        # self.model.key = api_key
        self.model = ChatOpenAI(
            model="gpt-3.5-turbo-16k",
            openai_api_key=api_key,
            temperature=self.temperature,
        )

    def inital_prompt(self) -> str:
        messages = [
            SystemMessage(content=self.system_def),
            HumanMessage(
                content=f"""
Write the first introductory part of a book of the type "chose your own adventure" \
where the reader, which is the main character, is a 17 year old girl coder. 
The introductory part is typically 3 pages long.
{self.general_formatting_style}
- End the text with {self.end_of_first_part}
"""
            ),
        ]
        return messages

    def next_prompt(self) -> str:
        messages = [
            SystemMessage(content=self.system_def),
            HumanMessage(
                content=f"""
Write the paagraph that follows the choice of the user of the book of the \
type "chose your own adventure".
{self.general_formatting_style}
- End the text with a chouce of 2 or 3 options for the reader, \
offering to turn to various pages.
"""
            ),
        ]
        return messages

    def format_text(self, input_text, width=50):
        return textwrap.fill(input_text, width=width)

    async def start(self):
        response = self.model(self.inital_prompt())
        formatted_text = self.format_text(response.content)
        print(formatted_text)
        input("Press Enter to continue...")
        return formatted_text

    async def step(self):
        response = self.model(self.next_prompt())
        formatted_text = self.format_text(response.content)
        print(formatted_text)
        choice = input("Make your choice...")
        return formatted_text

    def alive(self):
        return True


async def test():
    import cyoa_game.config as config

    adventure = Adventure(config.OpenAIKeyManager.get_api_key())
    text = await adventure.start()
    while adventure.alive():
        text = await adventure.step()


if __name__ == '__main__':
    asyncio.run(test())
