from langchain.llms.ollama import Ollama
from langchain.schema import StrOutputParser
from langchain.prompts import PromptTemplate


class OllamaMarkdownWriter(object):
    """Defines an Ollama LangChain Markdown Writer Prompted Model."""

    def __init__(
        self,
        ollama_model_name: str,
    ) -> None:
        """
        Initialize the Ollama Markdown Writer.

        Params:
            ollama_model_name (str): The Ollama model name to use.
        """
        self.ollama_model_name = ollama_model_name
        self.llm = Ollama(model=self.ollama_model_name, temperature=0)

    def write_section(self, instructions: str) -> str:
        """
        Write Section According to Instructions.

        Params:
            instructions (str): Instructions for rewriting.

        Returns:
            str: Rewritten section.
        """
        write_template = PromptTemplate.from_template(
            template="""
                You are an AI tasked with programmatically writing a section of a document according to a specification.
                You are in a code pipeline, and you are given the section to write and instructions for how to write it.
                Any text you output will be taken as the written section exactly and inserted into the document downstream.
                You will be a reliable and trusted part of the pipeline, only outputting as told to do so.
                Stick as closely to the instructions as possible given the section to write.
                Please be concise and to the point, only writing what is necessary to fulfill the instructions.
                Note that any Math equations should be written in LaTeX surrounded by $ signs.
                The section must have a header representative of the section (starting with #).
                The section must be written in Markdown.

                The instructions are: "{inst}"
                Your final written output: """,
        )
        chain = write_template | self.llm | StrOutputParser()
        output = chain.invoke({"inst": instructions})
        output = output.strip()  # only strip spaces from the ends
        output += "\n\n"  # add a newline to the end
        return output
