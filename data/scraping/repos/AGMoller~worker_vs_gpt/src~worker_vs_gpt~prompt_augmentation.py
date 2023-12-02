import json
import os
import time
from typing import Callable, Dict, List, Tuple


from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv 
import pandas as pd

from tqdm import tqdm

import hydra

from worker_vs_gpt.prompting.langchain_prompting import (
    DataTemplates,
)
from worker_vs_gpt.config import (
    HATE_SPEECH_DATA_DIR,
    SENTIMENT_DATA_DIR,
    TEN_DIM_DATA_DIR,
    ANALYSE_TAL_DATA_DIR,
    CROWDFLOWER_DATA_DIR,
    EMPATHY_DATA_DIR,
    POLITENESS_DATA_DIR,
    HYPO_DATA_DIR,
    INTIMACY_DATA_DIR,
    SAMESIDE_DATA_DIR,
    TALKDOWN_DATA_DIR,
    AugmentConfig,
    LORA_WEIGHTS_DIR,
)

from worker_vs_gpt.utils import balanced_sample_df, parse_output, rng, get_pipeline

load_dotenv()


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="config_prompt_augmentation.yaml",
)
def main(cfg: AugmentConfig) -> None:
    augmentation_templates = DataTemplates()

    # Load data and template
    if cfg.dataset == "analyse-tal":
        raise NotImplementedError
    elif cfg.dataset == "hate-speech":
        # read json
        text = "tweet"  # text column
        HS_dict = {"OFF": "offensive", "NOT": "not offensive"}
        dataset = pd.read_json(os.path.join(HATE_SPEECH_DATA_DIR, "base.json"))
        augmentation_prompt = augmentation_templates.get_hate_speech_prompt()
    elif cfg.dataset == "sentiment":
        text = "text"  # text column
        dataset = pd.read_json(os.path.join(SENTIMENT_DATA_DIR, "base.json"))
        augmentation_prompt = augmentation_templates.get_sentiment_prompt()
    elif cfg.dataset == "ten-dim":
        text = "h_text"  # text column (can be text or h_text)
        dataset = pd.read_json(os.path.join(TEN_DIM_DATA_DIR, "base.json"))
        augmentation_prompt = augmentation_templates.get_ten_dim_prompt()
        label_to_description = {
            "knowledge": "Exchange of ideas or information",
            "power": "Having power over the behavior and outcomes of another",
            "respect": "Conferring status, appreciation, gratitude, or admiration upon another",
            "trust": "Will of relying on the actions or judgments of another",
            "social_support": "Giving emotional or practical aid and companionship",
            "similarity_identity": "Shared interests, motivations, outlooks or Shared sense of belonging to the same community or group",
            "fun": "Experiencing leisure, laughter, and joy",
            "conflict": "Contrast or diverging views",
            "neutral": "neutral communication",
        }
    elif cfg.dataset == "crowdflower":
        text = "text"  # text column
        dataset = pd.read_json(os.path.join(CROWDFLOWER_DATA_DIR, "base.json"))
        augmentation_prompt = augmentation_templates.get_crowdfl_prompt()
    elif cfg.dataset == "empathy#empathy_bin":
        text = "text"  # text column
        dataset = pd.read_json(os.path.join(EMPATHY_DATA_DIR, "base.json"))
        augmentation_prompt = augmentation_templates.get_empathy_prompt()
    elif cfg.dataset == "hayati-politeness":
        text = "text"  # text column
        dataset = pd.read_json(os.path.join(POLITENESS_DATA_DIR, "base.json"))
        augmentation_prompt = augmentation_templates.get_politeness_prompt()
    elif cfg.dataset == "hypo-l":
        text = "text"  # text column
        dataset = pd.read_json(os.path.join(HYPO_DATA_DIR, "base.json"))
        augmentation_prompt = augmentation_templates.get_hypo_prompt()
    elif cfg.dataset == "questionintimacy":
        text = "text"  # text column
        dataset = pd.read_json(os.path.join(INTIMACY_DATA_DIR, "base.json"))
        augmentation_prompt = augmentation_templates.get_intimacy_prompt()
    elif cfg.dataset == "same-side-pairs":
        text = "text"  # text column
        dataset = pd.read_json(os.path.join(SAMESIDE_DATA_DIR, "base.json"))
        augmentation_prompt = augmentation_templates.get_sameside_prompt()
    elif cfg.dataset == "talkdown-pairs":
        text = "text"  # text column
        dataset = pd.read_json(os.path.join(TALKDOWN_DATA_DIR, "base.json"))
        augmentation_prompt = augmentation_templates.get_talkdown_prompt()
    else:
        raise ValueError(f"Dataset not found: {cfg.dataset}")

    temperature = 0
    if cfg.sampling == "balanced":
        dataset = balanced_sample_df(dataset, len(dataset))
        temperature = 1

        # get all duplicate rows
        duplicateRowsDF = dataset[dataset.duplicated([text])]

    df = pd.DataFrame(columns=[f"{text}", "target", f"augmented_{text}"])

    for idx, input_text in tqdm(dataset[text].items()):
        # Refresh the model
        llm = ChatOpenAI(model_name=cfg.model, temperature=temperature)

        llm_chain = LLMChain(prompt=augmentation_prompt, llm=llm)

        if cfg.dataset == "ten-dim":
            description = label_to_description[dataset["target"][idx]]
            try:
                output = llm_chain.run(
                    {
                        "text": input_text,
                        "social_dimension": dataset["target"][idx],
                        "social_dimension_description": description,
                    }
                )
            except Exception as e:
                print(e)
                print(f"Error with text: {input_text}")
                print("-------")
                continue
        elif cfg.dataset == "sentiment":
            try:
                output = llm_chain.run(
                    {
                        "text": input_text,
                        "sentiment": dataset["target"][idx],
                    }
                )
            except Exception as e:
                print(e)
                print(f"Error with text: {input_text}")
                print("-------")
                continue
        elif cfg.dataset == "hate-speech":
            label = HS_dict[dataset["target"][idx]]
            try:
                output = llm_chain.run(
                    {
                        "text": input_text,
                        "hate_speech": label,
                    }
                )
            except Exception as e:
                print(e)
                print(f"Error with {input_text}")
                print("-------")
                continue
        elif cfg.dataset == "analyse-tal":
            label = AT_dict[dataset["target"][idx]]
            try:
                output = llm_chain.run(
                    {
                        "text": input_text,
                        "language": language,
                        "label": label,
                    }
                )
            except Exception as e:
                print(e)
                print(f"Error with {input_text}")
                print("-------")
                continue
        elif cfg.dataset == "crowdflower":
            try:
                output = llm_chain.run(
                    {
                        "text": input_text,
                        "emotion": dataset["target"][idx],
                    }
                )
            except Exception as e:
                print(e)
                print(f"Error with text: {input_text}")
                print("-------")
                continue
        elif cfg.dataset == "empathy#empathy_bin":
            try:
                output = llm_chain.run(
                    {
                        "text": input_text,
                        "empathy": dataset["target"][idx],
                    }
                )
            except Exception as e:
                print(e)
                print(f"Error with text: {input_text}")
                print("-------")
                continue
        elif cfg.dataset == "hayati-politeness":
            try:
                output = llm_chain.run(
                    {
                        "text": input_text,
                        "politeness": dataset["target"][idx],
                    }
                )
            except Exception as e:
                print(e)
                print(f"Error with text: {input_text}")
                print("-------")
                continue
        elif cfg.dataset == "hypo-l":
            try:
                output = llm_chain.run(
                    {
                        "text": input_text,
                        "hypo": dataset["target"][idx],
                    }
                )
            except Exception as e:
                print(e)
                print(f"Error with text: {input_text}")
                print("-------")
                continue
        elif cfg.dataset == "questionintimacy":
            try:
                output = llm_chain.run(
                    {
                        "text": input_text,
                        "intimacy": dataset["target"][idx],
                    }
                )
            except Exception as e:
                print(e)
                print(f"Error with text: {input_text}")
                print("-------")
                continue
        elif cfg.dataset == "same-side-pairs":
            try:
                output = llm_chain.run(
                    {
                        "text": input_text,
                        "side": dataset["target"][idx],
                    }
                )
            except Exception as e:
                print(e)
                print(f"Error with text: {input_text}")
                print("-------")
                continue
        elif cfg.dataset == "talkdown-pairs":
            try:
                output = llm_chain.run(
                    {
                        "text": input_text,
                        "talkdown": dataset["target"][idx],
                    }
                )
            except Exception as e:
                print(e)
                print(f"Error with text: {input_text}")
                print("-------")
                continue
        else:
            raise NotImplementedError

        augmented_text = parse_output(input_string=output)
        pl = pd.DataFrame(augmented_text, columns=[f"augmented_{text}"])
        pl[text] = input_text
        pl["target"] = dataset["target"][idx]
        df = df.append(
            pl,
            ignore_index=True,
        ) # type: ignore

    df = df.sample(frac=1, random_state=rng).reset_index(drop=True)

    if cfg.dataset == "analyse-tal":
        raise NotImplementedError
    elif cfg.dataset == "hate-speech":
        df.to_json(
            HATE_SPEECH_DATA_DIR / f"{cfg.sampling}_{cfg.model}_augmented.json",
            orient="records",
        )
    elif cfg.dataset == "sentiment":
        df.to_json(
            SENTIMENT_DATA_DIR / f"{cfg.sampling}_{cfg.model}_augmented.json",
            orient="records",
        )
    elif cfg.dataset == "ten-dim":
        df.to_json(
            TEN_DIM_DATA_DIR / f"{cfg.sampling}_{cfg.model}_augmented.json",
            orient="records",
        )
    elif cfg.dataset == "crowdflower":
        df.to_json(
            CROWDFLOWER_DATA_DIR / f"{cfg.sampling}_{cfg.model}_augmented.json",
            orient="records",
        )
    elif cfg.dataset == "empathy#empathy_bin":
        df.to_json(
            EMPATHY_DATA_DIR / f"{cfg.sampling}_{cfg.model}_augmented.json",
            orient="records",
        )
    elif cfg.dataset == "hayati-politeness":
        df.to_json(
            POLITENESS_DATA_DIR / f"{cfg.sampling}_{cfg.model}_augmented.json",
            orient="records",
        )
    elif cfg.dataset == "hypo-l":
        df.to_json(
            HYPO_DATA_DIR / f"{cfg.sampling}_{cfg.model}_augmented.json",
            orient="records",
        )
    elif cfg.dataset == "questionintimacy":
        df.to_json(
            INTIMACY_DATA_DIR / f"{cfg.sampling}_{cfg.model}_augmented.json",
            orient="records",
        )
    elif cfg.dataset == "same-side-pairs":
        df.to_json(
            SAMESIDE_DATA_DIR / f"{cfg.sampling}_{cfg.model}_augmented.json",
            orient="records",
        )
    elif cfg.dataset == "talkdown-pairs":
        df.to_json(
            TALKDOWN_DATA_DIR / f"{cfg.sampling}_{cfg.model}_augmented.json",
            orient="records",
        )



if __name__ == "__main__":
    main() # type: ignore
