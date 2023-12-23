from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    PromptTemplate,
)

# Prompt templates
system_template_script = PromptTemplate(
    input_variables=["output_format", "html_content"],
    template="""You are a helpful assitant that helps people create python scripts for web scraping.
    --------------------------------
    The example of the html content is: {html_content}
    --------------------------------
    You have to create a python function that extract information from an html code using web scrapping.
    Try to select the deeper class that is common among the elements to make de find_all function.

    Your answer SHOULD only contain the python function code without any aditional word or character.

    Import the used libraries above the function definition.

    The function name must be extract_info.

    The function have to receive the html data as a parameter.

    Your function needs to extract information for all the elements with similar attributes.

    An element could have missing attributes

    Before calling .text or ['href'] methods, check if the element exists.

    ----------------
    FINAL ANSWER EXAMPLE:
    from bs4 import BeautifulSoup

    def extract_info(html):
        ...CODE...
        return {output_format}
    ----------------
    
    Always check if the element exists before calling some method.

    """,
)

human_template_script = PromptTemplate(input_variables=[], template="give me the code")

# Chat Prompt objects
system_template_script_prompt = SystemMessagePromptTemplate.from_template(
    system_template_script.template
)
human_template_script_prompt = HumanMessagePromptTemplate.from_template(
    human_template_script.template
)
chat_script_prompt = ChatPromptTemplate.from_messages(
    [system_template_script_prompt, human_template_script_prompt]
)
