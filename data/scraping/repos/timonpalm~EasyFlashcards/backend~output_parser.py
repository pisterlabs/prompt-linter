import re
from langchain.output_parsers import ListOutputParser
from typing import List

class OutputParser(ListOutputParser):
    """Parse a numbered list."""

    def get_format_instructions(self) -> str:
        return (
            "Use the following format:"
            "flashcards:"
            "1. [front]: content of frontside [back]: content of backside"
            "2. [front]: content of frontside [back]: content of backside"
            "3. ..."
        )

    def parse(self, text: str) -> List[str]:
        """Parse the output of an LLM call."""

        finished_cards = []

        pattern = r"\d+\.\s([^\n]+)"

        # Extract the text of each item
        cards = re.findall(pattern, text)

        for card in cards:
            pattern = r"\[front\]: (.*?) \[back\]: (.*)"

            # Use pattern to search in the string to extract
            match = re.search(pattern, card)

            if match:
                finished_cards.append((match.group(1), match.group(2)))
            else:
                print("No match found!")

        return finished_cards

    @property
    def _type(self) -> str:
        return "numbered-list"