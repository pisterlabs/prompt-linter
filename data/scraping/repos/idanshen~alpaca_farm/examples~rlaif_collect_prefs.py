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

BASE_PREAMBLE = "You are an expert summary rater. Given a piece of text and\
two of its possible summaries, output 1 or 2 to indicate\
which summary is better."

OPENAI_PREAMBLE = "A good summary is a shorter piece of text that has the\
essence of the original. It tries to accomplish the same\
purpose and conveys the key information from the original\
post. Below we define four evaluation axes for summary\
quality: coherence, accuracy, coverage, and overall quality.\
Coherence: This axis answers the question “how coherent is\
the summary on its own?” A summary is coherent if it’s easy\
to understand when read on its own and free of English errors.\
A summary is not coherent if it’s difficult to understand\
what the summary is trying to say. Generally, it’s more\
important that the summary is understandable than it being\
free of grammar errors.\n\
Accuracy: This axis answers the question “does the factual\
information in the summary accurately match the post?” A\
summary is accurate if it doesn’t say things that aren’t in\
the article, it doesn’t mix up people, and generally is not\
misleading.\n\
Coverage: This axis answers the question “how well does\
the summary cover the important information in the post?” A\
summary has good coverage if it mentions the main information\
from the post that’s important to understand the situation\
described in the post. A summary has poor coverage if\
someone reading only the summary would be missing several\
important pieces of information about the situation in the\
post. A summary with good coverage should also match the\
purpose of the original post (e.g. to ask for advice).\n\
Overall quality: This axis answers the question “how good\
is the summary overall at representing the post?” This can\
encompass all of the above axes of quality, as well as others\
you feel are important. If it’s hard to find ways to make\
the summary better, the overall quality is good. If there\
are lots of different ways the summary can be made better,\
the overall quality is bad.\n\
You are an expert summary rater. Given a piece of text and\
two of its possible summaries, output 1 or 2 to indicate\
which summary best adheres to coherence, accuracy, coverage,\
and overall quality as defined above."


@dataclass
class DataArguments:
    dataset_path: str = field(default="openai/summarize_from_feedback")
    dataset_name: str = field(default="comparisons") 
    partition: str = field(default='train')
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
        labels = []
        for j in range(2):
            max_tokens = model_args.max_tokens
            text = row['info']['post']
            summary1 = row['summaries'][0]['text']
            summary2 = row['summaries'][1]['text']
            
            # flip summary order if j==1
            if j == 1:
                summary3 = summary1
                summary1 = summary2
                summary2 = summary3
            while True:
                try:
                    response = openai.Completion.create(
                        model=model_args.model,
                        prompt=f"{BASE_PREAMBLE}\n{OPENAI_PREAMBLE}\nText - {text}\nSummary 1 - {summary1}\nSummary 2 - {summary2}\nPreferred Summary= ",
                        max_tokens=max_tokens,
                        logprobs=model_args.logprobs,
                        temperature=0.0,
                    )
                    break
                except openai.error.OpenAIError as e:
                    logging.warning(f'OpenAIError: {e}.')
                    if "Please reduce your prompt" in str(e):
                        max_tokens = int(max_tokens * 0.8)
                        logging.warning(f"Reducing target length to {max_tokens}, Retrying...")
                        if max_tokens == 0:
                            logging.exception("Prompt is already longer than max context length. Error:")
                            raise e
                except Exception as e:
                    raise e
            
            tokens = response['choices'][0]['logprobs']['tokens']
            indices = [i for i, x in enumerate(tokens) if x == "1" or x == "2"]
            
            if len(indices) == 0:
                print("No indices found!")
                continue
            elif len(indices) > 1:
                print("More than one index found!")
                continue
            
            i = indices[0]
            logprobs = response['choices'][0]['logprobs']['top_logprobs'][i]
            
            if '1' not in logprobs or '2' not in logprobs:
                print("No 1 or 2 in logprobs!")
                continue
            
            label = [logprobs['1'], logprobs['2']] if j == 0 else [logprobs['2'], logprobs['1']]
            labels.append(label)
        
        if len(labels) != 2:
            print('Skipping row due to 1 or more logprobs not including the right tokens')
            continue
        
        labels = [(labels[0][0] + labels[1][0])/2, (labels[0][1] + labels[1][1])/2]
        labels = softmax(labels)
        
        if compute_alignment and cur_idx in val_indices:
            llm_label = np.argmax(labels)
            if llm_label == row['choice']:
                total_correct += 1
            total += 1 
            
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
    val_indices = get_val_indices(dataset, data_args.eval_ratio)
    
    # evaluate
    save_filepath = os.path.join(data_args.results_dir ,f'rlaif_{model_args.model}_{data_args.dataset_path}_data_{data_args.partition}.json')
    results, alignment_score = evaluate_dataset(dataset, model_args, save_filepath=save_filepath, compute_alignment=True, val_indices=val_indices)
    print(f"Alignment score: {alignment_score}")
    
    # save results
    jdump(results, save_filepath)


if __name__ == "__main__":
    main()