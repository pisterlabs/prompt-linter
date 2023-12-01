import json
import pickle
from dataclasses import dataclass

from tqdm import tqdm

from moca.adapter import GPT3Adapter, HuggingfaceAdapter, DelphiAdapter, ClaudeAdapter, TogetherChatAdapter, \
    TogetherCompletionAdapter
from moca.data import CausalDataset, MoralDataset, Example, AbstractDataset, JsonSerializable
from moca.evaluator import AccuracyEvaluatorWithAmbiguity, CorrelationEvaluator, RMSEEvaluator, AuROCEvaluator, \
    MAEEvaluator, CEEvaluator
from moca.prompt import CausalJudgmentPrompt, MoralJudgmentPrompt, JudgmentPrompt

from typing import Union
from pathlib import Path


@dataclass
class ExperimentResult(JsonSerializable):
    acc: float
    conf_interval: tuple[float, float]
    r: float
    p: float
    rmse: float
    mae: float
    auroc: float
    ce: float


def run_template_for_huggingface(cd: AbstractDataset, adapter: HuggingfaceAdapter,
                                 jp: JudgmentPrompt, method: str = 'yesno', batch_size: int = 4):
    batched_instances = []
    all_instances, all_choice_scores, all_label_indices = [], [], []
    ex: Example
    for ex in tqdm(cd):
        instance = jp.apply(ex)
        batched_instances.append(instance)

        all_label_indices.append(ex.answer_dist)
        all_instances.append(instance)

        if len(batched_instances) == batch_size:
            choice_scores = adapter.adapt(batched_instances, method=method)
            all_choice_scores.extend(choice_scores)
            batched_instances = []

    if len(batched_instances) > 0:
        choice_scores = adapter.adapt(batched_instances, method=method)
        all_choice_scores.extend(choice_scores)

    return all_instances, all_choice_scores, all_label_indices


def exp1_causal_huggingface(model_name: str = "bert-base-uncased", batch_size: int = 4):
    if model_name in ['roberta-large', 'albert-xxlarge-v2',
                      'gpt2-xl', "EleutherAI/gpt-neo-1.3B"]:
        adapter = HuggingfaceAdapter(model_name=model_name, device='cpu')
    else:
        adapter = HuggingfaceAdapter(model_name=model_name, device='cpu')

    cd = CausalDataset()

    evaluator = AccuracyEvaluatorWithAmbiguity()
    corr_evaluator = CorrelationEvaluator()
    rmse_evaluator = RMSEEvaluator()
    mae_evaluator = MAEEvaluator()
    auroc_evaluator = AuROCEvaluator()
    ce_evaluator = CEEvaluator()

    all_instances, all_choice_scores, all_label_indices = [], [], []

    instances, choice_scores, label_indices = run_template_for_huggingface(cd, adapter, CausalJudgmentPrompt(
        "./prompts/exp1_causal_prompt.jinja"),
                                                                           method='yesno', batch_size=batch_size)
    all_choice_scores.extend(choice_scores)
    all_label_indices.extend(label_indices)
    all_instances.extend(instances)

    instances, choice_scores, label_indices = run_template_for_huggingface(cd, adapter, CausalJudgmentPrompt(
        "./prompts/exp1_causal_prompt_2.jinja"),
                                                                           method='multiple_choice',
                                                                           batch_size=batch_size)
    all_choice_scores.extend(choice_scores)
    all_label_indices.extend(label_indices)
    all_instances.extend(instances)

    acc, conf_interval = evaluator.evaluate(all_choice_scores, all_label_indices)
    r, p = corr_evaluator.evaluate(all_choice_scores, all_label_indices)
    rmse = rmse_evaluator.evaluate(all_choice_scores, all_label_indices)
    mae = mae_evaluator.evaluate(all_choice_scores, all_label_indices)
    auroc = auroc_evaluator.evaluate(all_choice_scores, all_label_indices)
    ce = ce_evaluator.evaluate(all_choice_scores, all_label_indices)

    with open(f"../../results/preds/exp1_{model_name.replace('-', '_').replace('/', '_')}_causal_preds.pkl", "wb") as f:
        pickle.dump((all_instances, all_choice_scores, all_label_indices), f)

    print()
    print("Model: ", model_name)
    print(f"Causal 3-class Accuracy: {acc:.4f} ({conf_interval[0]:.4f}, {conf_interval[1]:.4f})")
    print(f"Causal Correlation: {r:.4f} (p={p:.4f})")
    print(f"Causal RMSE: {rmse:.4f}")
    print(f"Causal MAE: {mae:.4f}")
    print(f"Causal AuROC: {auroc:.4f}")
    print(f"Causal CE: {ce:.4f}")

    return ExperimentResult(acc, conf_interval, r, p, rmse, mae, auroc, ce)


def exp1_moral_huggingface(model_name: str = "bert-base-uncased", batch_size: int = 4):
    adapter = HuggingfaceAdapter(model_name=model_name)
    cd = MoralDataset()

    evaluator = AccuracyEvaluatorWithAmbiguity()
    corr_evaluator = CorrelationEvaluator()
    rmse_evaluator = RMSEEvaluator()
    mae_evaluator = MAEEvaluator()
    auroc_evaluator = AuROCEvaluator()
    ce_evaluator = CEEvaluator()

    all_instances, all_choice_scores, all_label_indices = [], [], []

    instances, choice_scores, label_indices = run_template_for_huggingface(cd, adapter, MoralJudgmentPrompt(
        "./prompts/exp1_moral_prompt.jinja"), method='yesno', batch_size=batch_size)
    all_choice_scores.extend(choice_scores)
    all_label_indices.extend(label_indices)
    all_instances.extend(instances)

    instances, choice_scores, label_indices = run_template_for_huggingface(cd, adapter, MoralJudgmentPrompt(
        "./prompts/exp1_moral_prompt_2.jinja"), method='multiple_choice', batch_size=batch_size)
    all_choice_scores.extend(choice_scores)
    all_label_indices.extend(label_indices)
    all_instances.extend(instances)

    with open(f"../../results/preds/exp1_{model_name.replace('-', '_').replace('/', '_')}_moral_preds.pkl", "wb") as f:
        pickle.dump((all_instances, all_choice_scores, all_label_indices), f)

    acc, conf_interval = evaluator.evaluate(all_choice_scores, all_label_indices)
    r, p = corr_evaluator.evaluate(all_choice_scores, all_label_indices)
    rmse = rmse_evaluator.evaluate(all_choice_scores, all_label_indices)
    mae = mae_evaluator.evaluate(all_choice_scores, all_label_indices)
    auroc = auroc_evaluator.evaluate(all_choice_scores, all_label_indices)
    ce = ce_evaluator.evaluate(all_choice_scores, all_label_indices)

    # Moral 3-class Accuracy: 0.2742 (0.1631, 0.3852)
    print()
    print("Model: ", model_name)
    print(f"Moral 3-class Accuracy: {acc:.4f} ({conf_interval[0]:.4f}, {conf_interval[1]:.4f})")
    print(f"Moral Correlation: {r:.4f} (p={p:.4f})")
    print(f"Moral RMSE: {rmse:.4f}")
    print(f"Moral MAE: {mae:.4f}")
    print(f"Moral AuROC: {auroc:.4f}")
    print(f"Moral CE: {ce:.4f}")

    return ExperimentResult(acc, conf_interval, r, p, rmse, mae, auroc, ce)


def exp1_moral_delphi():
    # we can't use prompt here because that's not how Delphi works
    # Delphi shouldn't have confidence intervals, because we assigned fake probability

    adapter = DelphiAdapter()
    cd = MoralDataset()

    evaluator = AccuracyEvaluatorWithAmbiguity()

    all_choice_scores, all_label_indices = [], []
    ex: Example
    for ex in tqdm(cd):
        choice_scores = adapter.adapt(ex.story + " " + ex.question, method='yesno')
        all_choice_scores.append(choice_scores)
        all_label_indices.append(ex.answer_dist)

    acc, conf_interval = evaluator.evaluate(all_choice_scores, all_label_indices)

    print()
    print("Delphi")
    print(f"Moral 3-class Accuracy: {acc:.4f} ({conf_interval[0]:.4f}, {conf_interval[1]:.4f})")

    return ExperimentResult(acc, conf_interval, 0, 0, 0, 0, 0, 0)


def run_template_for_gpt3(cd: AbstractDataset, adapter: Union[GPT3Adapter, ClaudeAdapter],
                          jp: JudgmentPrompt, method: str = 'yesno'):
    all_instances, all_choice_scores, all_label_dist = [], [], []
    for ex in tqdm(cd):
        instance = jp.apply(ex)
        choice_scores = adapter.adapt(instance, method=method)
        all_choice_scores.append(choice_scores)
        all_label_dist.append(ex.answer_dist)
        all_instances.append(instance)

    return all_instances, all_choice_scores, all_label_dist


#  credential_file="./credential.txt"
def exp1_causal(engine: str = 'text-davinci-002', chat_mode=False, together=False):
    cd = CausalDataset()

    evaluator = AccuracyEvaluatorWithAmbiguity()
    corr_evaluator = CorrelationEvaluator()
    rmse_evaluator = RMSEEvaluator()
    mae_evaluator = MAEEvaluator()
    auroc_evaluator = AuROCEvaluator()
    ce_evaluator = CEEvaluator()

    if not together:
        if 'claude' not in engine:
            adapter = GPT3Adapter(engine=engine)
        else:
            adapter = ClaudeAdapter(engine=engine)
    else:
        if chat_mode:
            adapter = TogetherChatAdapter(engine=engine)
        else:
            adapter = TogetherCompletionAdapter(engine=engine)

    all_instances, all_choice_scores, all_label_indices = [], [], []

    if not together and chat_mode:
        # only chat model from OpenAI we can't get logprob
        prompt_file = "../moca/prompts/exp1_causal_chat_prompt.jinja"
    else:
        prompt_file = "../moca/prompts/exp1_causal_prompt.jinja"

    instances, choice_scores, label_indices = run_template_for_gpt3(cd, adapter, CausalJudgmentPrompt(prompt_file),
                                                                    method='yesno')
    all_choice_scores.extend(choice_scores)
    all_label_indices.extend(label_indices)
    all_instances.extend(instances)

    if not together and chat_mode:
        # only chat model from OpenAI we can't get logprob
        prompt_file = "../moca/prompts/exp1_causal_chat_prompt_2.jinja"
    else:
        prompt_file = "../moca/prompts/exp1_causal_prompt_2.jinja"

    instances, choice_scores, label_indices = run_template_for_gpt3(cd, adapter, CausalJudgmentPrompt(prompt_file),
                                                                    method='multiple_choice')
    all_choice_scores.extend(choice_scores)
    all_label_indices.extend(label_indices)
    all_instances.extend(instances)

    acc, conf_interval = evaluator.evaluate(all_choice_scores, all_label_indices)
    r, p = corr_evaluator.evaluate(all_choice_scores, all_label_indices)
    rmse = rmse_evaluator.evaluate(all_choice_scores, all_label_indices)
    mae = mae_evaluator.evaluate(all_choice_scores, all_label_indices)
    auroc = auroc_evaluator.evaluate(all_choice_scores, all_label_indices)
    ce = ce_evaluator.evaluate(all_choice_scores, all_label_indices)

    # create the folder if it doesn't exist
    Path("results/preds").mkdir(parents=True, exist_ok=True)

    with open(f"results/preds/exp1_{engine}_causal_preds.pkl", "wb") as f:
        pickle.dump((all_instances, all_choice_scores, all_label_indices), f)

    print()
    print(f"engine: {engine}")
    print(f"Causal Accuracy: {acc:.4f} ({conf_interval[0]:.4f}, {conf_interval[1]:.4f})")
    print(f"Causal Correlation: {r:.4f} (p={p:.4f})")
    print(f"Causal RMSE: {rmse:.4f}")
    print(f"Causal MAE: {mae:.4f}")
    print(f"Causal AuROC: {auroc:.4f}")
    print(f"Causal CE: {ce:.4f}")

    return ExperimentResult(acc, conf_interval, r, p, rmse, mae, auroc, ce)

def exp1_moral(engine: str = 'text-davinci-002', chat_mode=False, together=False):
    md = MoralDataset()

    evaluator = AccuracyEvaluatorWithAmbiguity()
    corr_evaluator = CorrelationEvaluator()
    rmse_evaluator = RMSEEvaluator()
    mae_evaluator = MAEEvaluator()
    auroc_evaluator = AuROCEvaluator()
    ce_evaluator = CEEvaluator()

    if not together:
        if 'claude' not in engine:
            adapter = GPT3Adapter(engine=engine)
        else:
            adapter = ClaudeAdapter(engine=engine)
    else:
        if chat_mode:
            adapter = TogetherChatAdapter(engine=engine)
        else:
            adapter = TogetherCompletionAdapter(engine=engine)

    all_instances, all_choice_scores, all_label_indices = [], [], []

    if not together and chat_mode:
        # only chat model from OpenAI we can't get logprob
        prompt_file = "../moca/prompts/exp1_moral_chat_prompt.jinja"
    else:
        prompt_file = "../moca/prompts/exp1_moral_prompt.jinja"

    instances, choice_scores, label_indices = run_template_for_gpt3(md, adapter, MoralJudgmentPrompt(prompt_file),
                                                                    method='yesno')
    all_choice_scores.extend(choice_scores)
    all_label_indices.extend(label_indices)
    all_instances.extend(instances)

    if not together and chat_mode:
        # only chat model from OpenAI we can't get logprob
        prompt_file = "../moca/prompts/exp1_moral_chat_prompt_2.jinja"
    else:
        prompt_file = "../moca/prompts/exp1_moral_prompt_2.jinja"

    instances, choice_scores, label_indices = run_template_for_gpt3(md, adapter,
                                                                    MoralJudgmentPrompt(prompt_file),
                                                                    method='multiple_choice')
    all_choice_scores.extend(choice_scores)
    all_label_indices.extend(label_indices)
    all_instances.extend(instances)

    acc, conf_interval = evaluator.evaluate(all_choice_scores, all_label_indices)
    r, p = corr_evaluator.evaluate(all_choice_scores, all_label_indices)
    rmse = rmse_evaluator.evaluate(all_choice_scores, all_label_indices)
    mae = mae_evaluator.evaluate(all_choice_scores, all_label_indices)
    auroc = auroc_evaluator.evaluate(all_choice_scores, all_label_indices)
    ce = ce_evaluator.evaluate(all_choice_scores, all_label_indices)

    Path("results/preds").mkdir(parents=True, exist_ok=True)
    with open(f"results/preds/exp1_{engine}_moral_preds.pkl", "wb") as f:
        pickle.dump((all_instances, all_choice_scores, all_label_indices), f)

    print()
    print(f"engine={engine}")
    print(f"Moral Accuracy: {acc:.4f} ({conf_interval[0]:.4f}, {conf_interval[1]:.4f})")
    print(f"Moral Correlation: {r:.4f} (p={p:.4f})")
    print(f"Moral RMSE: {rmse:.4f}")
    print(f"Moral MAE: {mae:.4f}")
    print(f"Moral AuROC: {auroc:.4f}")
    print(f"Moral CE: {ce:.4f}")

    return ExperimentResult(acc, conf_interval, r, p, rmse, mae, auroc, ce)


def produce_table1():
    result = {}

    # causal
    for model_name in ['google/electra-large-generator', 'bert-base-uncased', 'bert-large-uncased', 'roberta-large',
                       'albert-xxlarge-v2', 'gpt2-xl']:
        er = exp1_causal_huggingface(model_name=model_name, batch_size=32)
        result[model_name] = er.json

    for engine in ["text-babbage-001", 'text-curie-001', 'text-davinci-002']:
        er = exp1_causal(engine=engine)
        result[engine] = er.json

    Path("results").mkdir(parents=True, exist_ok=True)
    json.dump(result, open('results/exp1_causal_full_result.json', 'w'), indent=2)

    # moral
    result = {}
    for model_name in ['roberta-large', 'google/electra-large-generator', 'bert-base-uncased', 'bert-large-uncased',
                       'albert-xxlarge-v2', 'gpt2-xl']:
        er = exp1_moral_huggingface(model_name=model_name)
        result[model_name] = er.json

    er = exp1_moral_delphi()
    result['delphi'] = er.json

    for engine in ["text-babbage-001", 'text-curie-001', 'text-davinci-002']:
        er = exp1_moral(engine=engine)
        result[engine] = er.json

    Path("results").mkdir(parents=True, exist_ok=True)
    json.dump(result, open('results/exp1_moral_full_result.json', 'w'), indent=2)


def run_openai_models():
    result = {}

    for engine in ["text-babbage-001", 'text-curie-001', 'text-davinci-002']:
        er = exp1_causal(engine=engine)
        result[engine] = er.json

    Path("results").mkdir(parents=True, exist_ok=True)
    json.dump(result, open('results/exp1_causal_full_result.json', 'w'), indent=2)

    # moral
    result = {}

    Path("results").mkdir(parents=True, exist_ok=True)
    for engine in ["text-babbage-001", 'text-curie-001', 'text-davinci-002']:
        er = exp1_moral(engine=engine)
        result[engine] = er.json

    json.dump(result, open('results/exp1_moral_full_result.json', 'w'), indent=2)


def run_latest_openai():
    result = {}
    for engine in ['text-davinci-003']:
        er = exp1_causal(engine=engine)
        result[engine] = er.json

    Path("results").mkdir(parents=True, exist_ok=True)
    json.dump(result, open('results/exp1_openai_causal_last_result.json', 'w'), indent=2)

    result = {}

    for engine in ['text-davinci-003']:
        er = exp1_moral(engine=engine)
        result[engine] = er.json

    Path("results").mkdir(parents=True, exist_ok=True)
    json.dump(result, open('results/exp1_openai_moral_last_result.json', 'w'), indent=2)


def run_latest_chat_models():
    result = {}
    for engine in ['gpt-3.5-turbo-0613', 'gpt-4-0613']:
        er = exp1_causal(engine=engine, chat_mode=True)
        result[engine] = er.json

    Path("results").mkdir(parents=True, exist_ok=True)
    json.dump(result, open('results/exp1_openai_chatgpt_causal_result.json', 'w'), indent=2)

    result = {}

    for engine in ['gpt-3.5-turbo-0613', 'gpt-4-0613']:
        er = exp1_moral(engine=engine, chat_mode=True)
        result[engine] = er.json

    Path("results").mkdir(parents=True, exist_ok=True)
    json.dump(result, open('results/exp1_openai_chatgpt_moral_result.json', 'w'), indent=2)


def run_together_chat_models():
    result = {}
    for engine in ['llama-2-70b-chat']:
        er = exp1_causal(engine=engine, chat_mode=True, together=True)
        result[engine] = er.json

    Path("results").mkdir(parents=True, exist_ok=True)
    json.dump(result, open('results/exp1_together_causal_result.json', 'w'), indent=2)

    result = {}
    for engine in ['llama-2-70b-chat']:
        er = exp1_moral(engine=engine, chat_mode=True, together=True)
        result[engine] = er.json

    Path("results").mkdir(parents=True, exist_ok=True)
    json.dump(result, open('results/exp1_together_moral_result.json', 'w'), indent=2)


if __name__ == '__main__':
    pass
    # exp1_causal()
    # exp1_moral()

    # produce_table1()
    # run_latest_chat_models()

    # run_together_chat_models()
    # run_latest_openai()