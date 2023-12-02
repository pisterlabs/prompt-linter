import argparse
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
from misc.utils import print_colored_output


class PLM(LLM):
    """ Pig Latin LLM --- But not really an LLM, just a custom LLM implementation """
    additional_vowels: str

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        return " ".join([self.pig_latin(x) for x in prompt.split(" ")])

    def pig_latin(self, word):
        """ Convert a word to pig latin """

        # If the word starts with a vowel, add "way" to the end
        vowels = "aeiouAEIOU"+self.additional_vowels
        if word[0] in vowels:
            return word + "way"

        # If the word starts with a consonant, move the first letter to the end and add "ay"
        else:
            for i in range(len(word)):
                if word[i] in vowels:
                    return word[i:] + word[:i] + "ay"

        # If the word is only consonants, just add "ay" to the end
        return word + "ay"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"additional_vowels": self.additional_vowels}


def llm_interaction(additional_vowels=""):
    """ Script to interact with our custom PigLatin model --> PLM

    Args:
        additional_vowels (str): Additional vowels to add to the model

    Returns:
        None; interacts with the user and prints the user inputs and model responses to the console
    """

    # Initialize the ChatOpenAI instance with the specified parameters
    llm = PLM(additional_vowels=additional_vowels)

    # Continue the conversation loop until the user decides to exit
    while True:

        # Get the user input and break if the user decides to exit
        input_text = input("\nEnter your message (or press enter to exit):\n")
        if not input_text: break

        # response = llm.generate([input_text]).generations[0][0].text
        response = llm(input_text)

        print_colored_output(input_text, response_text=response.strip(), full_color=True)


def main():
    parser = argparse.ArgumentParser(description="Generate a LLM model via LangChain OpenAI module")
    parser.add_argument("-av", "--additional_vowels", type=str, default="", help="Additional vowels for pig latin.")
    args = parser.parse_args()

    # Call the main function
    llm_interaction(**args.__dict__)


if __name__ == "__main__":
    main()
