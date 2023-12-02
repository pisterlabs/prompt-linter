from langchain.llms.ollama import Ollama
from langchain.schema import Document, StrOutputParser
from langchain.prompts import PromptTemplate


class OllamaMarkdownRewriter(object):
    """Defines an Ollama LangChain Markdown Rewriter Prompted Model."""

    def __init__(
        self,
        section_to_rewrite: Document,
        ollama_model_name: str,
    ) -> None:
        """
        Initialize the Ollama Markdown Rewriter.

        Params:
            section_to_rewrite (Document): The section to rewrite.
            ollama_model_name (str): The Ollama model name to use.
        """
        self.section_to_rewrite = section_to_rewrite
        self.ollama_model_name = ollama_model_name
        self.llm = Ollama(model=self.ollama_model_name, temperature=0)

    def rewrite_section(self, instructions: str) -> str:
        """
        Rewrite Section According to Instructions.

        Params:
            instructions (str): Instructions for rewriting.

        Returns:
            str: Rewritten section.
        """
        rewrite_template = PromptTemplate.from_template(
            template="""
                You are an AI tasked with programmatically rewriting a section of a document according to a specification.
                You are in a code pipeline, and you are given the section to rewrite and instructions for how to rewrite it.
                Any text you output will be taken as the rewritten section exactly and inserted into the document downstream.
                You will be a reliable and trusted part of the pipeline, only outputting as told to do so.
                Stick as closely to the instructions as possible given the section to rewrite.
                Please be concise and to the point, only writing what is necessary to fulfill the instructions.
                Note that any Math equations should be written in LaTeX surrounded by $ signs.
                The section must have a header representative of the section (starting with #).
                The section must be written in Markdown.

                The section to rewrite is: "{section_to_rewrite}"
                The instructions are: "{inst}"
                Your final rewritten output: """,
            partial_variables={
                "section_to_rewrite": self.section_to_rewrite.page_content,
            },
        )
        chain = rewrite_template | self.llm | StrOutputParser()
        output = chain.invoke({"inst": instructions})
        output = output.strip()  # only strip spaces from the ends
        output += "\n\n"  # add a newline to the end
        return output
