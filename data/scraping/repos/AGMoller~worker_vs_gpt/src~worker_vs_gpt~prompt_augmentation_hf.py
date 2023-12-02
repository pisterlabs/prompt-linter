import json
import logging
import os
import time
from typing import Callable, Dict, List, Tuple


from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import pandas as pd
import sys

from tqdm import tqdm

import hydra
from worker_vs_gpt.prompting.datasets_config import (
    HatespeechConfig,
    SentimentConfig,
    TenDimConfig,
    CrowdflowerConfig,
    SameSidePairsConfig,
    HayatiPolitenessConfig,
    HypoConfig,
    EmpathyConfig,
    QuestionIntimacyConfig,
    TalkdownPairsConfig,
)

from worker_vs_gpt.prompting.langchain_prompting import (
    DataTemplates,
)
from worker_vs_gpt.prompting.huggingface_prompting import HuggingfaceChatTemplate
from worker_vs_gpt.config import (
    HATE_SPEECH_DATA_DIR,
    SENTIMENT_DATA_DIR,
    TEN_DIM_DATA_DIR,
    CROWDFLOWER_DATA_DIR,
    EMPATHY_DATA_DIR,
    POLITENESS_DATA_DIR,
    HYPO_DATA_DIR,
    INTIMACY_DATA_DIR,
    SAMESIDE_DATA_DIR,
    TALKDOWN_DATA_DIR,
    AugmentConfig,
    LORA_WEIGHTS_DIR,
    PromptConfig,
    HF_HUB_MODELS,
    LOGS_DIR,
)

from worker_vs_gpt.utils import (
    balanced_sample_df,
    parse_output,
    rng,
    get_pipeline,
    parse_llama_output,
)

load_dotenv()


def setup_logging(cfg):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=f"{str(LOGS_DIR)}/augmentation/{cfg.dataset}_{cfg.model}.log",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s]: %(message)s",
        filemode="w",
    )


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="config_prompt_augmentation.yaml",
)
def main(cfg: AugmentConfig) -> None:
    setup_logging(cfg)

    if len(sys.argv) > 1:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = sys.argv[1]

    # Load data and template
    if cfg.dataset == "analyse-tal":
        raise NotImplementedError
    elif cfg.dataset == "hate-speech":
        # read json
        text = "tweet"  # text column
        HS_dict = {"OFF": "offensive", "NOT": "not offensive"}
        dataset = pd.read_json(os.path.join(HATE_SPEECH_DATA_DIR, "base.json"))
        prompt_config = HatespeechConfig
    elif cfg.dataset == "sentiment":
        text = "text"  # text column
        dataset = pd.read_json(os.path.join(SENTIMENT_DATA_DIR, "base.json"))
        prompt_config = SentimentConfig
    elif cfg.dataset == "ten-dim":
        text = "h_text"  # text column (can be text or h_text)
        dataset = pd.read_json(os.path.join(TEN_DIM_DATA_DIR, "base.json"))
        prompt_config = TenDimConfig
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
        prompt_config = CrowdflowerConfig
    elif cfg.dataset == "empathy#empathy_bin":
        text = "text"  # text column
        dataset = pd.read_json(os.path.join(EMPATHY_DATA_DIR, "base.json"))
        prompt_config = EmpathyConfig
    elif cfg.dataset == "hayati-politeness":
        text = "text"  # text column
        dataset = pd.read_json(os.path.join(POLITENESS_DATA_DIR, "base.json"))
        prompt_config = HayatiPolitenessConfig
    elif cfg.dataset == "hypo-l":
        text = "text"  # text column
        dataset = pd.read_json(os.path.join(HYPO_DATA_DIR, "base.json"))
        prompt_config = HypoConfig
    elif cfg.dataset == "questionintimacy":
        text = "text"  # text column
        dataset = pd.read_json(os.path.join(INTIMACY_DATA_DIR, "base.json"))
        prompt_config = QuestionIntimacyConfig
    elif cfg.dataset == "same-side-pairs":
        text = "text"  # text column
        dataset = pd.read_json(os.path.join(SAMESIDE_DATA_DIR, "base.json"))
        prompt_config = SameSidePairsConfig
    elif cfg.dataset == "talkdown-pairs":
        text = "text"  # text column
        dataset = pd.read_json(os.path.join(TALKDOWN_DATA_DIR, "base.json"))
        prompt_config = TalkdownPairsConfig
    else:
        raise ValueError(f"Dataset not found: {cfg.dataset}")

    temperature = 0
    if cfg.sampling == "balanced":
        dataset = balanced_sample_df(dataset, len(dataset))
        temperature = 1

        # get all duplicate rows
        duplicateRowsDF = dataset[dataset.duplicated([text])]

    df = pd.DataFrame(columns=[f"{text}", "target", f"augmented_{text}"])

    template = HuggingfaceChatTemplate(
        model_name=HF_HUB_MODELS[cfg.model],
    ).get_template_augmentation(
        system_prompt=prompt_config.augmentation_system_prompt,
        task=prompt_config.augmentation_task_prompt,
    )

    for idx, input_text in tqdm(dataset[text].items()):
        # Refresh the model

        llm = InferenceClient(
            model=HF_HUB_MODELS[cfg.model],
            token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
        )

        has_output: bool = False

        while not has_output:
            if cfg.dataset == "ten-dim":
                description = label_to_description[dataset["target"][idx]]
                try:
                    output = llm.text_generation(
                        template.format(
                            text=input_text,
                            label=dataset["target"][idx],
                            social_dimension_description=description,
                        ),
                        max_new_tokens=2048,
                        temperature=temperature,
                        repetition_penalty=1.2,
                    )
                except Exception as e:
                    logging.info(e)
                    logging.info(f"Error with text: {input_text}")
                    logging.info("-------")
                    continue

                has_output = True
            else:
                try:
                    output = llm.text_generation(
                        template.format(text=input_text, label=dataset["target"][idx]),
                        max_new_tokens=2048,
                        temperature=temperature,
                        repetition_penalty=1.2,
                        return_full_text=False,
                        truncate=4096,
                    )
                except Exception as e:
                    logging.info(e)
                    logging.info(f"Error with text: {input_text}")
                    logging.info("-------")
                    continue
                has_output = True

        augmented_text = parse_llama_output(input_string=output)
        pl = pd.DataFrame(augmented_text, columns=[f"augmented_{text}"])
        pl[text] = input_text
        pl["target"] = dataset["target"][idx]
        df = df.append(
            pl,
            ignore_index=True,
        )  # type: ignore

        has_output = False

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
    main()  # type: ignore
