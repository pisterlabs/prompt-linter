from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import os
import pandas as pd
import numpy as np

from brics_crossmap.utils import helper as helper
from brics_crossmap.data_dictionary.curation import cur_logger, log, copy_log
from brics_crossmap.data_dictionary.curation.utils import curation_functions as cur
from brics_crossmap.data_dictionary.curation.chains.utils.format_qa import (
    ConceptQA,
    filter_df_for_qa,
    create_qa_prompt_from_df,
)
from brics_crossmap.data_dictionary.curation.chains.system_prompts.curator import (
    system_message_curator,
    example_qa,
)
from brics_crossmap.data_dictionary.curation.chains.utils.token_functions import (
    num_tokens_from_string,
    openaipricing,
)

from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.callbacks import get_openai_callback

from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.chat_models import ChatOpenAI

LANGCHAIN_TRACING_V2 = True
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
LANGCHAIN_API_KEY = os.environ["LANGCHAIN_API_KEY"]
LANGCHAIN_PROJECT = "ddcuimap"


def llm_qa_multiple_choice(cfg=None, **kwargs):
    if cfg is None:
        cfg = helper.load_config(helper.choose_file("Load config file from Step 1"))

    # LOAD OPENAI API KEY
    load_dotenv(find_dotenv())
    try:
        OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    except KeyError:
        raise Exception(
            "Could not load OpenAI API key. Make sure it's set in your environment variables."
        )

    fp = "C:/Users/armengolkm/Desktop/Full Pipeline Test v1.1.0/LLM_tests/DE_Step-1_Hydra-search (1)/curation/curation_semantic-search_crossencoder.csv"
    df_cur = pd.read_csv(fp, dtype="object")

    # CREATE OUTPUT DIRECTORY
    dir_curation = helper.create_folder(Path(fp).parent.joinpath("llm_curation"))

    # PREPROCESSING    #TODO: Fix this in prior pipeline so this can be removed
    df_cur["cross_encoder_score"] = df_cur["cross_encoder_score"].astype(float).round(3)
    df_cur["semantic_search_rank"] = df_cur["semantic_search_rank"].astype(int)

    # CREATE QA PROMPTS
    df_cur_filtered = []
    qa_prompts = []
    overall_rank = cfg.curation.preprocessing.overall_rank
    top_k_score = cfg.curation.preprocessing.top_k_score
    variables = df_cur[cfg.curation.preprocessing.variable_column].unique()
    for variable in variables:
        print(variable)

        df_filtered = (
            df_cur[df_cur["variable name"] == variable]
            .groupby("pipeline_name")
            .apply(lambda x: x[x["semantic_search_rank"] <= overall_rank])
            .drop_duplicates(subset=["data element concept identifiers"])
            .nlargest(top_k_score, "cross_encoder_score")
        )
        df_cur_filtered.append(df_filtered)
        qa_prompt = create_qa_prompt_from_df(df_filtered)
        qa_prompts.append(qa_prompt)

    # SET UP LANGCHAIN
    model_name = cfg.curation.langchain.openai.model_name
    encoding_name = cfg.curation.langchain.openai.encoding_name
    temperature = cfg.curation.langchain.openai.temperature
    llm = OpenAI(model_name=model_name, temperature=temperature)
    output_parser = PydanticOutputParser(pydantic_object=ConceptQA)

    prompt = PromptTemplate(
        template=system_message_curator,
        input_variables=["qa_prompt", "example_qa"],
        partial_variables={
            "format_instructions": output_parser.get_format_instructions()
        },
    )

    inputs = []
    total_cost_prompts = 0
    for p in qa_prompts:
        _input = prompt.format_prompt(qa_prompt=p, example_qa=example_qa)
        num_tokens = num_tokens_from_string(
            _input.to_string(), encoding_name=encoding_name
        )
        cost = round(
            openaipricing(
                num_tokens, cfg.curation.langchain.openai.pricing_model, chatgpt=True
            ),
            3,
        )
        total_cost_prompts += cost
        print(f"# of tokens: {num_tokens} ({cost} USD)")
        inputs.append(_input)
    print(f"Total cost of prompts: {total_cost_prompts} USD")

    outputs = []
    with get_openai_callback() as cb:
        for i in inputs:
            output = llm(i.to_string())
            print(cb)
            try:
                answers = output_parser.parse(output).Answers
            except:
                new_parser = OutputFixingParser.from_llm(
                    parser=output_parser, llm=ChatOpenAI()
                )
                parsed_output = new_parser.parse(output)
                answers = parsed_output.Answers
            print(answers)
            outputs.append(answers)

    for df in df_cur_filtered:
        df["Reasoning"] = np.nan
        df["Confidence"] = np.nan
        df["keep"] = ""  # Add a 'keep' column filled with empty strings

    # After you get the model's output...
    for i, answers in enumerate(outputs):
        df = df_cur_filtered[i]
        for answer, details in answers.items():
            # Convert the answer number back to an index
            index = int(answer) - 1
            # Annotate the dataframe with the model's reasoning and confidence
            df.loc[df.index[index], "Reasoning"] = details.Reasoning
            df.loc[df.index[index], "Confidence"] = details.Confidence
            df.loc[df.index[index], "keep"] = 1  # Mark this row with an 1 to keep it

    df_final = pd.concat(df_cur_filtered)
    df_final.to_csv(
        dir_curation.joinpath(
            f"{cfg.custom.curation_settings.file_settings.directory_prefix}_curation.csv"
        ),
        index=False,
    )
    # TODO: merge llm curated results back with original
    # df_final.set_index('original_index', inplace=True)
    helper.save_config(cfg, dir_curation, "config_curation.yaml")

    return df_final, cfg


if __name__ == "__main__":
    # cfg = helper.load_config(helper.choose_file("Load config file from Step 1"))

    cfg = helper.compose_config(
        config_path="../configs/",
        config_name="config",
        overrides=[],
    )
    df_final, cfg = llm_qa_multiple_choice(cfg)
