from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import (ChatPromptTemplate,
                               HumanMessagePromptTemplate,
                               SystemMessagePromptTemplate)
from langchain.schema import SystemMessage


def language_structure():
    return "1,en-US,4,sv,5,es,6,ar,7,fr,8,yo,9,de,10,ot,11,ta,12,pt-BR"


def item_create_prompt(parser: PydanticOutputParser):
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a helpful assistant that generate relevant Item"),
        SystemMessage(content=f"Language ids: {language_structure()}"),
        SystemMessage(content=parser.get_format_instructions()),
        HumanMessagePromptTemplate.from_template("""Your job is to provide a final item from the following given prompt: 

        Prompt: {prompt}

        Please provide a response item that follows the output schema in the prompt language.""")
    ])


def item_follow_up_prompt(parser: PydanticOutputParser):
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a helpful assistant that provide a relevant item that follows up from a given item."),
        SystemMessage(content=f"Language ids: {language_structure()}"),
        SystemMessage(content=parser.get_format_instructions()),
        HumanMessagePromptTemplate.from_template("""Your job is to provide a final item from the following item answer the question:

        Item: {source}

        Question: {prompt}

        Please provide a response item that follows the output schema and maintains coherence by avoiding repetition while building upon the given item.""")
    ])


def item_regenerate_prompt(parser: PydanticOutputParser):
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a helpful assistant that regenerates provided content"),
        SystemMessage(content=f"Language ids: {language_structure()}"),
        SystemMessage(content=parser.get_format_instructions()),
        HumanMessagePromptTemplate.from_template("""Your job is to regenerate from the given Item provided below:

        Item: {source}

        Please provide a response item that follows the output schema.""")
    ])


def item_translate_prompt(parser: PydanticOutputParser):
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a helpful assistant that translates {source_language} Item into {output_language}"),
        SystemMessage(content=f"Language ids: {language_structure()}"),
        SystemMessage(content=parser.get_format_instructions()),
        HumanMessagePromptTemplate.from_template("""Your job is to produce a final [Item]
        by translating the following {source_language} into {output_language}:

        {source}

        Please provide a response item that follows the output schema.""")
    ])


def item_title_prompt(parser: PydanticOutputParser):
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a helpful assistant that generates relevant Item title for a given text"),
        SystemMessage(content=f"Language ids: {language_structure()}"),
        SystemMessage(content=parser.get_format_instructions()),
        HumanMessagePromptTemplate.from_template("""Your job is to produce a final [Item] from the following:

        {source}

        Please provide a response item that follows the output schema.""")
    ])
