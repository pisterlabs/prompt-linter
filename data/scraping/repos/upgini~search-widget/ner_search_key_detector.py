import logging
from typing import Tuple
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
import openai
import pandas as pd
import time
import re
import streamlit as st

from pandas.api.types import is_string_dtype


def get_template_text():
    template_text = (
        "You are a Named-entity recognition (NER) system. "
        "You take column name as first_input and values in this column as second_input. "
        "Your task is to recognize and extract specific name entity "
        "in the given column name and values and classify into a set of following predefined entity types: "
        "PERSON, PERSON_TYPE, "
        "ORGANIZATION, EVENT, "
        "PRODUCT, SKILL, QUANTITY, "
        "LOCATION, ADDRESS, POSTAL_CODE, "
        "EMAIL, HASHED_EMAIL, "
        "IP, URL, DOMAIN, PHONE_NUMBER, "
        "DATE, DATETIME, TIMEZONE, "
        "USER_AGENT, COMMENT, OTHER. "
        "Your output must consist of only one word. "
        "First_input: {col_name}. "
        "Second_input: {values}. "
        "Output:"
    )

    return template_text


def get_entity_gpt(col_name, values, logger, model_name="gpt-3.5-turbo", num_retries=10, default_entity="OTHER"):
    retries = 0
    entity = default_entity
    while retries < num_retries:
        try:
            model = ChatOpenAI(model_name=model_name, temperature=0)
            template_text = get_template_text()
            prompt_template = PromptTemplate(template=template_text, input_variables=["col_name", "values"])
            llm_chain = LLMChain(prompt=prompt_template, llm=model)
            entity = llm_chain.run({"col_name": col_name, "values": values})
            entity = re.sub(r"[^A-Z\s_]", "", entity.strip().upper()).split()
            if len(entity) > 0:
                return entity[0]
            else:
                return default_entity
        except Exception:
            logger.exception("Failed to invoke ChatOpenAI")
            retries += 1
            time.sleep(0.5)
            logger.info(f"Retrying embedding for {col_name}... ({retries}/{num_retries})")

    return default_entity


@st.cache_data(show_spinner=False)
def auto_detect_search_keys(
    df: pd.DataFrame,
    api_key: str,
    max_value_length=30,
    ner_threshold=30,
    _logger=logging.root,
) -> Tuple[dict, dict]:
    openai.api_key = api_key
    DEFAULT_ENTITY = "OTHER"

    search_keys_candidates = dict()
    generate_features_candiates = dict()
    try:
        for col in df.columns:
            values = [v[:max_value_length] if isinstance(v, str) else v for v in df[col].unique()[:ner_threshold]]
            try:
                _logger.info(f"Check column {col} by gpt-ner")
                entity = get_entity_gpt(col, values, _logger, default_entity=DEFAULT_ENTITY)
                _logger.info(f"Entity: {entity}")

                if entity in [
                    "POSTAL_CODE",
                    "EMAIL",
                    "HASHED_EMAIL",
                    "IP",
                    "PHONE_NUMBER",
                    "DATE",
                    "DATETIME",
                ]:
                    search_keys_candidates[col] = entity

                if is_string_dtype(df[col]) and entity in [
                    "PERSON",
                    "PERSON_TYPE",
                    "ORGANIZATION",
                    "EVENT",
                    "DESCRIPTION",
                    "COMMENT",
                    "PRODUCT",
                    "PRODUCT_CATEGORY",
                    "PRODUCT_DESCRIPTION",
                    "PRODUCT_COMMENT," "SKILL",
                    "JOB_TITLE",
                    "QUANTITY",
                    "ADDRESS_COMMENT",
                    "OTHER",
                ]:
                    generate_features_candiates[col] = entity

                # check condition
                # ner_passed = entity not in bad_entities_list
            except Exception:
                _logger.exception(f"Failed to call gpt ner for column {col}")
    except Exception:
        _logger.exception("Failed to autodetect keys")

    return search_keys_candidates, generate_features_candiates
