import os
from dataclasses import dataclass, field
import logging
import json

from tqdm import tqdm
import numpy as np
from scipy.special import softmax
import transformers
from datasets import load_dataset
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

from alpaca_farm.utils import jdump


@dataclass
class DataArguments:
    dataset_path: str = field(default="openai/summarize_from_feedback")
    dataset_name: str = field(default="comparisons") 
    partition: str = field(default='validation')
    eval_ratio: float = field(
        default=0.15,
        metadata={"help": "Ratio of training to use for evaluation (AI Labeler Alignment)."},
    )
    seed: int = field(
        default=21,
        metadata={"help": "Random seed."},
    )
    results_dir: str = field(
        default="./rlaif_results",
        metadata={"help": "Directory to save results."},
    )

@dataclass
class ModelArguments:
    model: str = field(default="gpt-3.5-turbo-instruct")
    max_tokens: int = field(default=500)
    logprobs: int = field(default=5, metadata={"help": "Number of log probs to return (5 is maximum)"})

def evaluate_dataset(dataset, model_args, save_filepath, compute_alignment=False, val_indices=None):
    SAVE_INT = 100
    results = []
    total_correct = 0.0
    total = 0.0
    
    if compute_alignment:
        assert val_indices is not None, "val_indices must be provided if compute_alignment is True."

    cur_idx = 0
    for row in tqdm(dataset):
        labels = [0.0, 0.0]

        text = row['info']['post']
        summary1 = row['summaries'][0]['text']
        summary2 = row['summaries'][1]['text']
            
        result = {
            'text': row['info']['post'],
            'summary1': row['summaries'][0]['text'],
            'summary2': row['summaries'][1]['text'],
            'llm_label': labels,
            'choice': row['choice'],
            'model': model_args.model,
        }
        results.append(result)
        cur_idx += 1

        if cur_idx % SAVE_INT == 0:
            jdump(results, save_filepath)
            print(f"Saved itr {cur_idx} results to {save_filepath}")

    if compute_alignment:
        return results, total_correct/total
    else:
        return results

def get_val_indices(dataset, eval_ratio):
    # Get the total size of the dataset
    n = len(dataset)

    # Calculate the split index 
    split_index = int(n*(1-eval_ratio))

    # Shuffle the indices 
    shuffled_indices = np.random.permutation(n) 

    # # Split data
    # train_indices = shuffled_indices[:split_index]
    val_indices = shuffled_indices[split_index:]

    # train_dataset = dataset[train_indices] 
    # val_dataset = dataset[val_indices]
    
    return val_indices
    
def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    np.random.seed(data_args.seed)
    os.makedirs(data_args.results_dir, exist_ok=True)

    # load and split data
    dataset = load_dataset(data_args.dataset_path, data_args.dataset_name, split=data_args.partition)
    
    # evaluate
    save_filepath = os.path.join(data_args.results_dir ,f'rlaif_{model_args.model}_{data_args.dataset_path}_data_{data_args.partition}.json')
    results = evaluate_dataset(dataset, model_args, save_filepath=save_filepath, compute_alignment=False)
    # print(f"Alignment score: {alignment_score}")
    
    # save results
    jdump(results, save_filepath)


if __name__ == "__main__":
    main()