import json
import os
import time
from typing import Callable, Dict, List, Tuple
import logging

from langchain import PromptTemplate, LLMChain, HuggingFaceHub
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from tqdm import tqdm
import wandb

import hydra

from huggingface_hub import InferenceClient
from worker_vs_gpt.prompting.huggingface_prompting import HuggingfaceChatTemplate
from worker_vs_gpt.prompting.langchain_prompting import (
    DataTemplates,
    ClassificationTemplates,
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
    PromptConfig,
    HF_HUB_MODELS,
    LOGS_DIR,
)

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

from worker_vs_gpt.utils import LabelMatcher, few_shot_sampling

load_dotenv()


def setup_logging(cfg):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=f"{str(LOGS_DIR)}/classification/{cfg.dataset}_{cfg.model}_{cfg.few_shot}-shot_per_class_sampling:{cfg.per_class_sampling}.log",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s]: %(message)s",
        filemode="w",
    )


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="config_prompt_classification.yaml",
)
def main(cfg: PromptConfig) -> None:
    setup_logging(cfg)

    classification_templates = ClassificationTemplates()

    # Load data and template
    if cfg.dataset == "analyse-tal":
        raise NotImplementedError
    elif cfg.dataset == "hate-speech":
        # read json
        text = "tweet"  # text column
        dataset = pd.read_json(os.path.join(HATE_SPEECH_DATA_DIR, "test.json"))
        train = pd.read_json(os.path.join(HATE_SPEECH_DATA_DIR, "train.json"))
        classification_prompt = classification_templates.classify_hate_speech()
        prompt_config = HatespeechConfig
    elif cfg.dataset == "sentiment":
        text = "text"  # text column
        dataset = pd.read_json(os.path.join(SENTIMENT_DATA_DIR, "test.json"))
        train = pd.read_json(os.path.join(SENTIMENT_DATA_DIR, "train.json"))
        classification_prompt = classification_templates.classify_sentiment()
        prompt_config = SentimentConfig
    elif cfg.dataset == "ten-dim":
        text = "h_text"  # text column (can be text or h_text)
        dataset = pd.read_json(os.path.join(TEN_DIM_DATA_DIR, "test.json"))
        train = pd.read_json(os.path.join(TEN_DIM_DATA_DIR, "train.json"))
        classification_prompt = classification_templates.classify_ten_dim()
        prompt_config = TenDimConfig
    elif cfg.dataset == "crowdflower":
        text = "text"  # text column (can be text or h_text)
        dataset = pd.read_json(os.path.join(CROWDFLOWER_DATA_DIR, "test.json"))
        train = pd.read_json(os.path.join(CROWDFLOWER_DATA_DIR, "train.json"))
        classification_prompt = classification_templates.classify_crowdflower()
        prompt_config = CrowdflowerConfig
    elif cfg.dataset == "same-side-pairs":
        text = "text"  # text column (can be text or h_text)
        dataset = pd.read_json(os.path.join(SAMESIDE_DATA_DIR, "test.json"))
        train = pd.read_json(os.path.join(SAMESIDE_DATA_DIR, "train.json"))
        classification_prompt = classification_templates.classfify_same_side()
        prompt_config = SameSidePairsConfig
    elif cfg.dataset == "hayati_politeness":
        text = "text"  # text column (can be text or h_text)
        dataset = pd.read_json(os.path.join(POLITENESS_DATA_DIR, "test.json"))
        train = pd.read_json(os.path.join(POLITENESS_DATA_DIR, "train.json"))
        classification_prompt = classification_templates.classfify_hayati()
        prompt_config = HayatiPolitenessConfig
    elif cfg.dataset == "hypo-l":
        text = "text"  # text column (can be text or h_text)
        dataset = pd.read_json(os.path.join(HYPO_DATA_DIR, "test.json"))
        train = pd.read_json(os.path.join(HYPO_DATA_DIR, "train.json"))
        classification_prompt = classification_templates.classfify_hypo()
        prompt_config = HypoConfig
    elif cfg.dataset == "empathy#empathy_bin":
        text = "text"  # text column (can be text or h_text)
        dataset = pd.read_json(os.path.join(EMPATHY_DATA_DIR, "test.json"))
        train = pd.read_json(os.path.join(EMPATHY_DATA_DIR, "train.json"))
        classification_prompt = classification_templates.classfify_empathy()
        prompt_config = EmpathyConfig
    elif cfg.dataset == "questionintimacy":
        text = "text"  # text column (can be text or h_text)
        dataset = pd.read_json(os.path.join(INTIMACY_DATA_DIR, "test.json"))
        train = pd.read_json(os.path.join(INTIMACY_DATA_DIR, "train.json"))
        classification_prompt = classification_templates.classfify_intimacy()
        prompt_config = QuestionIntimacyConfig
    elif cfg.dataset == "talkdown-pairs":
        text = "text"  # text column (can be text or h_text)
        dataset = pd.read_json(os.path.join(TALKDOWN_DATA_DIR, "test.json"))
        train = pd.read_json(os.path.join(TALKDOWN_DATA_DIR, "train.json"))
        classification_prompt = classification_templates.classify_talkdown()
        prompt_config = TalkdownPairsConfig
    else:
        raise ValueError(f"Dataset not found: {cfg.dataset}")

    # Predict
    y_pred = []
    idx = 0
    # Evaluate
    y_true = dataset["target"].values
    # Get all unique labels
    labels = list(set(y_true))

    label_mathcer = LabelMatcher(labels=labels, task=cfg.dataset)

    print(labels)

    llm = InferenceClient(
        model=HF_HUB_MODELS[cfg.model],
        token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
    )

    template = HuggingfaceChatTemplate(
        model_name=HF_HUB_MODELS[cfg.model],
    ).get_template_classification(
        system_prompt=prompt_config.classification_system_prompt,
        task=prompt_config.classification_task_prompt,
    )

    for input_text in tqdm(dataset[text]):
        # Sometimes refresh the model

        has_output: bool = False

        few_shot_samples = few_shot_sampling(
            df=train, n=cfg.few_shot, per_class_sampling=cfg.per_class_sampling
        )

        while not has_output:
            if cfg.model == "gpt-4":
                llm = ChatOpenAI(model_name=cfg.model, temperature=0)
                llm_chain = LLMChain(
                    prompt=classification_prompt, llm=llm, verbose=False
                )

                output = llm_chain.run(
                    {"few_shot": few_shot_samples, "text": input_text}
                )
            else:
                try:
                    output = llm.text_generation(
                        template.format(
                            few_shot=few_shot_samples,
                            text=input_text,
                        ),
                        max_new_tokens=25,
                        temperature=0.001,
                        # do_sample=True,
                        # stop_sequences=["\n"],
                        repetition_penalty=1.2,
                        truncate=4096,
                    )
                except Exception as e:
                    logging.info(f'Error with input text: "{input_text}"')
                    logging.error(e)
                    time.sleep(5)
                    continue
                has_output = True

        pred = label_mathcer(output, input_text)
        pred_matches = label_mathcer.label_check(output)
        pred2 = output
        y_pred.append(pred_matches)
        logging.info(f"Input: {input_text}")
        logging.info(f"Raw Prediction: {pred2}")
        logging.info(f"Prediction: {pred}")
        logging.info(f"Prediction Matches: {pred_matches}")
        logging.info(f"True: {y_true[idx]}")
        logging.info("---" * 10)
        idx += 1

        has_output = False

    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", labels=labels)
    # roc_auc = roc_auc_score(y_true, y_probs, average="macro")
    report = classification_report(
        y_true=y_true, y_pred=y_pred, output_dict=True, labels=labels
    )

    logging.info(
        classification_report(
            y_true=y_true, y_pred=y_pred, output_dict=False, labels=labels
        )
    )

    # Initialize wandb
    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=f"{cfg.model}-{cfg.few_shot}-shot-per_class_sampling:{cfg.per_class_sampling}",
        group=f"{cfg.dataset}",
        tags=["prompt_classification"],
        config={
            "model": cfg.model,
            "dataset": cfg.dataset,
            "text_column": text,
            "few_shot": cfg.few_shot,
            "per_class_sampling": cfg.per_class_sampling,
        },
    )

    metrics = {"test/accuracy": accuracy, "test/f1": f1}

    # Log results
    wandb.log(
        metrics,
    )

    df = pd.DataFrame(report)
    df["metric"] = df.index
    table = wandb.Table(data=df)

    wandb.log(
        {
            "classification_report": table,
        }
    )

    wandb.finish()


if __name__ == "__main__":
    main()
