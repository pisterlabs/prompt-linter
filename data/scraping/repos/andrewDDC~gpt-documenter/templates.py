from langchain import PromptTemplate
from gpt_documenter.utils import dedent


def summary_json_template():
    return dedent(
        """
        {
        "functionallity_explanation_summary": "",
        "function_args": "",
        "explained_function_args": "",
        "function_output": ""
        }
        """)


def doc_function_template():
    return PromptTemplate(
        template=dedent(
            """
            You are a bot designed to generate documentation for software.
            Your job is to document a {language} function based on:
            -its own content
            -The summaries of all the functions that are used inside the main function.
            The generated summary has to follow this JSON template:
            {summary_json_template}
            You must ensure that the JSON is well formatted.
            Make sure to not add extra fields. Do not add the used_functions into the result.
            Make sure to not add trailing commas.

            The function you have to document is:
             
            {text}

            The summaries of the functions used inside are:
             
            {summaries}
             
            The summary's JSON result is:
            """
        ),
        input_variables=["text", "summaries", "language", "summary_json_template"],
    )


def doc_base_function_template():
    return PromptTemplate(
        template=dedent(
            """
            You are a bot designed to generate documentation for software.
            Your job is to document a {language} function based on its content.
            The generated summary has to follow this JSON template:
             
            {summary_json_template}
            You must ensure that the JSON is well formatted.
            Make sure to not add extra fields.
            Make sure to not add trailing commas.
        
            The function you have to document is:

            {text}

            The summary's JSON result is:
            """
        ),
        input_variables=["text", "language", "summary_json_template"],
    )
