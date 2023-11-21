from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


def get_author_chain():
    chat = ChatOpenAI(temperature=0.3, model_name='gpt-4',
                      request_timeout=200, max_retries=3, max_tokens=2000)

    system_template = "You are a helpful assistant that writes news articles. "
    system_message_prompt = SystemMessagePromptTemplate.from_template(
        system_template)

    human_message_prompt = HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=[
                                                      "summaries"], template="Summaries: {summaries}\n Write a medium-length article about the above summaries. Use markdown to format the text."))
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt])
    author_chain = LLMChain(llm=chat, prompt=chat_prompt,
                            output_key="article")
    return author_chain


def get_question_chain():

    template = """Article: {article}

    What followup question a reader could have about the article? Put each question on a new line. """

    prompt = PromptTemplate(template=template, input_variables=["article"])

    llm = OpenAI(temperature=0.5)

    question_chain = LLMChain(prompt=prompt, llm=llm, output_key="questions")

    return question_chain


def get_writer_chain():
    chat = ChatOpenAI(temperature=0.3, model_name='gpt-4',
                      request_timeout=200, max_retries=4, max_tokens=3000)

    system_template = "You are a helpful assistant that writes news articles. "
    system_message_prompt = SystemMessagePromptTemplate.from_template(
        system_template)

    human_message_prompt = HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=[
                                                      "extra_information", "article"], template="Additional Context:\n{extra_information}\nArticle Draft:\n{article}\nTask:\nUsing markdown format, create a medium lenfth article that seamlessly integrates the additional context provided with the existing draft. Ensure that the final article is coherent, engaging, and well-structured."))
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt])
    writer_chain = LLMChain(llm=chat, prompt=chat_prompt,
                            output_key="final_draft")
    return writer_chain


def get_editor_chain():
    chat = ChatOpenAI(temperature=0.3, model_name='gpt-4',
                      request_timeout=200, max_retries=4, max_tokens=6000)

    system_template = "You are a helpful assistant that writes news articles. "
    system_message_prompt = SystemMessagePromptTemplate.from_template(
        system_template)

    human_message_prompt = HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=[
                                                      "final_draft"], template="Article: {final_draft}\n Task: Using markdown add a TLDR section to the top of the article with bullet points."))
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt])
    editor_chain = LLMChain(llm=chat, prompt=chat_prompt,
                            output_key="tldr",
                            verbose=True)
    return editor_chain

def get_title_chain(article):
    chat = ChatOpenAI(temperature=0.3, model_name='gpt-3.5-turbo',
                      request_timeout=200, max_retries=4)

    system_template = "You are a helpful assistant that creates SEO title from user article. "
    system_message_prompt = SystemMessagePromptTemplate.from_template(
        system_template)

    human_message_prompt = HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template=f'Article: {article}\n Task: What is a good SEO title for the article. Only return the title. Do not put the title in quotes.'))
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt])
    title_chain = LLMChain(llm=chat, prompt=chat_prompt,
                            output_key="title",
                            verbose=True)
    return title_chain

def get_description_chain(article):
    chat = ChatOpenAI(temperature=0.3, model_name='gpt-3.5-turbo',
                      request_timeout=200, max_retries=4)

    system_template = "You are a helpful assistant that creates SEO description from user article. "
    system_message_prompt = SystemMessagePromptTemplate.from_template(
        system_template)

    human_message_prompt = HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template=f'Article: {article}\n Task: Return a good SEO description for the article. Only return the description.'))
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt])
    description_chain = LLMChain(llm=chat, prompt=chat_prompt,
                            output_key="description",
                            verbose=True)
    return description_chain
