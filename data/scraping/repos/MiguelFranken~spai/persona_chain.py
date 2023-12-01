from langchain.chains import TransformChain
from langchain.chains import SequentialChain

from transformers.persona_transformer import transform_clean, transform_name, transform_story


def create_clean_extra_spaces_chain():
    """
    Returns a TransformChain object that removes extra spaces from the input persona information.

    :return: TransformChain object
    """
    return TransformChain(
        input_variables=["persona"],
        output_variables=["clean_persona"],
        transform=transform_clean
    )


def create_persona_name_possessive_transformer_chain():
    """
    Returns a TransformChain object that transforms the clean_persona input variable into persona_name and persona_name_possessive output variables.

    :return: TransformChain object
    """
    return TransformChain(
        input_variables=["clean_persona"],
        output_variables=["persona_name", "persona_name_possessive"],
        transform=transform_name
    )


def create_persona_story_transformer_chain():
    """
    Returns a TransformChain object that transforms the clean_persona input variable into the persona_story output variable.

    :return: TransformChain object
    """
    return TransformChain(
        input_variables=["clean_persona"],
        output_variables=["persona_story"],
        transform=transform_story
    )


def create_transform_persona_chain():
    """
    Returns a SequentialChain object that applies a sequence of transformations to a persona dictionary, including cleaning
    extra spaces, extracting the persona name and its possessive form, and extracting the persona story.

    :return: SequentialChain object
    """
    clean_chain = create_clean_extra_spaces_chain()
    name_chain = create_persona_name_possessive_transformer_chain()
    story_chain = create_persona_story_transformer_chain()

    return SequentialChain(
        chains=[clean_chain, name_chain, story_chain],
        input_variables=["persona"],
        output_variables=["persona_story", "persona_name", "persona_name_possessive"],
        verbose=True
    )
