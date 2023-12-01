from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.chat_models import ChatOpenAI


def generate_vul_report(report):
    # Define a string template for a System role with two input variables: `output_language` and `max_words`
    system_template = """You are a Research Analyst, specialized in writing connected informative reports. \
    You will analyze question and answers given, and writes well-connected, actionable, and informative reports.
    """
    # Create a prompt template for a System role
    system_message_prompt_template = SystemMessagePromptTemplate.from_template(system_template)

    # Define a string template for a Human role with the `sample_text` input variable
    human_template = """Write vulnerability report following sections: description, vulnerability \ 
    details, affected products and impact, vulnerability identifiers threat actors using \
    vulnerability, exploitation details, mitigations, recommendations shor-term and long-term strategies \
    references focus only on available information using markdown format and try using available \
    information without losing. \n\nInterview Summary Information:{report_information} """

    # Create a prompt template for a Human role
    human_message_prompt_template = HumanMessagePromptTemplate.from_template(human_template)

    # Create a chat prompt template by combining the System and Human message templates
    chat_prompt_template = ChatPromptTemplate.from_messages(
        [system_message_prompt_template, human_message_prompt_template])

    # Generate a final prompt by passing all three variables (`output_language`, `max_words`, `sample_text`)
    final_prompt = chat_prompt_template.format_prompt(report_information=report).to_messages()
    chat = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
    completion = chat(final_prompt)
    return completion.content
