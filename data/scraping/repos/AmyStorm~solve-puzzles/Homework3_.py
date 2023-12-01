import collections
from contextlib import nullcontext
from collections import namedtuple
from datasets import load_dataset
import json
import numpy as np
import random
import re
import string
import torch
from typing import List

seed = 1

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


import openai
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
f = open('homework3_print.txt', 'w+')
sys.stdout = f

# import torch
#
# if torch.cuda.is_available():
#     !pip install faiss-gpu==1.7.0
# else:
#     !pip install faiss-cpu==1.7.0

import os
import sys
sys.path.insert(0, 'ColBERT/')

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Collection
from colbert.searcher import Searcher
from utility.utils.dpr import has_answer, DPR_normalize


index_name = "cs224u.collection.2bits"
index_home = os.path.join("experiments", "notebook", "indexes")

with Run().context(RunConfig(experiment='notebook')):
    searcher = Searcher(index=index_name)


def _find_generated_answer(tokens, newline="\n" ):
    """Our LMs tend to insert initial newline characters before
    they begin generating text. This function ensures that we
    properly capture the true first line as the answer while
    also ensuring that token probabilities are aligned."""
    answer_token_indices = []
    char_seen = False
    for i, tok in enumerate(tokens):
        # This is the main condition: a newline that isn't an initial
        # string of newlines:
        if tok == newline and char_seen:
            break
        # Keep the initial newlines for consistency:
        elif tok == newline and not char_seen:
            answer_token_indices.append(i)
        # Proper tokens:
        elif tok != newline:
            char_seen = True
            answer_token_indices.append(i)
    return answer_token_indices

# "gpt-neo-125M" "gpt-neo-1.3B" "gpt-neo-2.7B" "gpt-j-6B"
eleuther_model_name = "gpt-neo-125M"

eleuther_tokenizer = AutoTokenizer.from_pretrained(
    f"EleutherAI/{eleuther_model_name}",
    padding_side="left",
    padding='longest',
    truncation='longest_first', max_length=2000)
eleuther_tokenizer.pad_token = eleuther_tokenizer.eos_token

eleuther_model = AutoModelForCausalLM.from_pretrained(
    f"EleutherAI/{eleuther_model_name}")

device = "cuda" if torch.cuda.is_available() else "cpu"
eleuther_model = eleuther_model.to(device)


def run_eleuther(prompts, temperature=0.1, top_p=0.95, **generate_kwargs):
    """
    Parameters
    ----------
    prompts : iterable of str
    temperature : float
        It seems best to set it low for this task!
    top_p : float

    For options for `generate_kwargs`, see:

    https://huggingface.co/docs/transformers/master/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate

    Options that are likely to be especially relevant include
    `temperature`, `length_penalty`, and the parameters that
    determine the decoding strategy. With `num_return_sequences > 1`,
    the default parameters in this function do multinomial sampling.

    Returns
    -------
    list of dicts

    {"prompt": str,
     "generated_text": str, "generated_tokens": list of str, "generated_probs": list of float,
     "answer": str, "answer_tokens": list of str, "answer_probs": list of float
    }

    """
    prompt_ids = eleuther_tokenizer(
        prompts, return_tensors="pt", padding=True).input_ids.to(device)

    with torch.inference_mode():
        # Automatic mixed precision if possible.
        with torch.cuda.amp.autocast() if torch.cuda.is_available() else nullcontext():
            model_output = eleuther_model.generate(
                prompt_ids,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
                max_new_tokens=50,
                num_return_sequences=1,
                pad_token_id=eleuther_tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
                **generate_kwargs)

    # Converting output scores using the helpful recipe here:
    # https://discuss.huggingface.co/t/generation-probabilities-how-to-compute-probabilities-of-output-scores-for-gpt2/3175
    gen_ids = model_output.sequences[:, prompt_ids.shape[-1]:]
    gen_probs = torch.stack(model_output.scores, dim=1).softmax(-1)
    gen_probs = torch.gather(gen_probs, 2, gen_ids[:, :, None]).squeeze(-1)

    # Generated texts, including the prompts:
    gen_texts = eleuther_tokenizer.batch_decode(
        model_output.sequences, skip_special_tokens=True)

    data = []
    iterator = zip(prompts, gen_ids, gen_texts, gen_probs)
    for prompt, gen_id, gen_text, gen_prob in iterator:
        gen_tokens = eleuther_tokenizer.convert_ids_to_tokens(gen_id)
        generated_text = gen_text[len(prompt):]
        gen_prob = [float(x) for x in gen_prob.cpu().numpy()]  # float for JSON storage
        ans_indices = _find_generated_answer(gen_tokens, newline="Ċ")
        answer_tokens = [gen_tokens[i] for i in ans_indices]
        answer_probs = [gen_prob[i] for i in ans_indices]
        answer = "".join(answer_tokens).replace("Ġ", " ").replace("Ċ", "\n")
        data.append({
            "prompt": prompt,
            "generated_text": generated_text,
            "generated_tokens": gen_tokens,
            "generated_probs": gen_prob,
            "generated_answer": answer,
            "generated_answer_probs": answer_probs,
            "generated_answer_tokens": answer_tokens})

    return data


SquadExample = namedtuple("SquadExample",  "id title context question answers")


def get_squad_split(squad, split="validation"):
    """
    Use `split='train'` for the train split.

    Returns
    -------
    list of SquadExample named tuples with attributes
    id, title, context, question, answers

    """
    fields = squad[split].features
    data = zip(*[squad[split][field] for field in fields])
    return [SquadExample(eid, title, context, question, answers["text"])
            for eid, title, context, question, answers in data]


squad = load_dataset("squad")
squad_dev = get_squad_split(squad)

dev_exs = sorted(squad_dev, key=lambda x: hash(x.id))[: 200]




def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s: str) -> List[str]:
    """Normalize string and split string into tokens."""
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold: str, a_pred: str) -> int:
    """Compute the Exact Match score."""
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1_from_tokens(gold_toks: List[str], pred_toks: List[str]) -> float:
    """Compute the F1 score from tokenized gold answer and prediction."""
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_f1(a_gold: str, a_pred: str) -> float:
    """Compute the F1 score."""
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    return compute_f1_from_tokens(gold_toks, pred_toks)


def build_zero_shot_openqa_prompt(question, passage, joiner="\n\n"):
    title, context = passage.split(" | ", 1)
    segs = [
        f"Title: {title}",
        f"Background: {context}",
        f"Q: {question}",
        "A:"
    ]
    return joiner.join(segs)


def evaluate(examples, prompts, gens):
    """Generic evalution function.

    Parameters
    ----------
    examples: iterable of `SquadExample` instances
    prompts: list of str
    preds: list of LM-generated texts to evaluate as answers

    Returns
    -------
    dict with keys "em_per", "macro_f1", "examples", where
    each "examples" value is a dict

    """
    results = []
    for ex, prompt, gen in zip(examples, prompts, gens):
        answers = ex.answers
        pred = gen['generated_answer']
        # The result is the highest EM from the available answer strings:
        em = max([compute_exact(ans, pred) for ans in answers])
        f1 = max([compute_f1(ans, pred) for ans in answers])
        gen.update({
            "id": ex.id,
            "question": ex.question,
            "prediction": pred,
            "answers": answers,
            "em": em,
            "f1": f1
        })
        results.append(gen)
    data = {}
    data["macro_f1"] = np.mean([d['f1'] for d in results])
    data["em_per"] = sum([d['em'] for d in results]) / len(results)
    data["examples"] = results
    return data




def build_few_shot_open_qa_prompt(question, passage, train_exs, joiner="\n\n"):
    """Few-shot OpenQA prompts.

    Parameters
    ----------
    question : str
    passage : str
        Presumably something retrieved via search.
    train_exs : iterable of SQuAD train examples
        These can be obtained via a random sample from
        `squad_train` as defined above.
    joiner : str
        The character to use to join pieces of the prompt
        into a single str.

    Returns
    -------
    str, the prompt

    """
    segs = []
    title, context = passage.split(" | ", 1)
    for t in train_exs:
        segs += [
            f"Title: {t.title}",
            f"Background: {t.context}",
            f"Q: {t.question}",
            f"A: {t.answers[0]}"
        ]
    segs += [
        f"Title: {title}",
        f"Background: {context}",
        f"Q: {question}",
        "A:"
    ]
    return joiner.join(segs)



def get_passages_with_scores(question, k=5):
    """Pseudo-probabilities from the retriever.

    Parameters
    ----------
    question : str
    k : int
        Number of passages to retrieve.

    Returns
    -------
    passages (list of str), passage_probs (np.array)

    """
    # Use the `searcher` to get `k` passages for `questions`:
    passages = []
    passage_scores = []
    results = searcher.search(question, k)
    for passage_id, passage_rank, passage_score in zip(*results):
        passages += {searcher.collection[passage_id]}
        passage_scores += {passage_score}


    # Softmax normalize the scores and convert the list to
    # a NumPy array:
    total = sum(passage_scores)
    passage_probs = np.array([x/total for x in passage_scores])



    # Get the passages as a list of texts:
    return passages, passage_probs



def answer_scoring(passages, passage_probs, prompts, gen_func=run_eleuther):
    """Implements our basic scoring strategy.

    Parameters
    ----------
    passages : list of str
    passage_probs : list of float
    prompts : list of str
    gen_func : either `run_eleuther` or `run_gpt3`

    Returns
    -------
    list of pairs (score, dict), sorted with the largest score first.
    `dict` should be the return value of `gen_func` for an example.

    """
    data = []
    for passage, passage_prob, prompt in zip(passages, passage_probs, prompts):


        # Run `gen_func` on [prompt] (crucially, the singleton list here),
        # and get the dictionary `gen` from the singleton list `gen_func`
        # returns, and then use the values to score `gen` according to our
        # scoring method.
        #
        # Be sure to use "generated_answer_probs" for the scores.

        gen = gen_func(prompt)
        answer_probs = gen[0].get('generated_answer_probs')
        scores = [a * passage_prob for a in answer_probs]
        for score in scores:
            if ' I don\'t know.' in gen[0].get('generated_answer'):
                data.append([0, gen[0]])
            else:
                data.append([score, gen[0]])


    # Return `data`, sorted with the highest scoring `(score, gen)`
    # pair given first.
    data.sort(key=lambda x: x[0], reverse=True)
    return data


def answer_scoring_ev(passages, passage_probs, prompts, gen_func=run_eleuther):

    data = []
    for passage, passage_prob, prompt in zip(passages, passage_probs, prompts):


        # Run `gen_func` on [prompt] (crucially, the singleton list here),
        # and get the dictionary `gen` from the singleton list `gen_func`
        # returns, and then use the values to score `gen` according to our
        # scoring method.
        #
        # Be sure to use "generated_answer_probs" for the scores.

        gen = gen_func(prompt)
        answer_probs = gen[0].get('generated_answer_probs')
        scores = [a * passage_prob for a in answer_probs]
        for score in scores:
            if ' I don\'t know.' in gen[0].get('generated_answer'):
                data.append([0, gen[0], prompt])
            else:
                data.append([score, gen[0], prompt])


    # Return `data`, sorted with the highest scoring `(score, gen)`
    # pair given first.
    data.sort(key=lambda x: x[0], reverse=True)
    return data[0][2], data[0][1]


def answer_exec(question):
    """Example usage for answer_scoring. Here we extract the top-scoring
    results, which can then be used in an evaluation."""
    passages, passage_probs = get_passages_with_scores(question)
    train_exs = []
    for squadexample in dev_exs:
        if squadexample.question == question or squadexample.title in question:
            train_exs.append(squadexample)
    prompts = [build_few_shot_open_qa_prompt(question, psg, train_exs) for psg in passages]
    data = answer_scoring(passages, passage_probs, prompts)
    #print('question ended:' + question + '---answer:' + data[0][1].get('generated_answer'))
    # Top-scoring answer string:
    return data[0][1]



def create_bakeoff_submission():
    filename = os.path.join("data", "openqa", "cs224u-openqa-test-unlabeled.txt")

    # This should become a mapping from questions (str) to response
    # dicts from your system.
    gens = {}

    with open(filename) as f:
        questions = f.read().splitlines()

    # `questions` is the list of questions you need to evaluate your system on.
    # Put whatever code you need to in here to evaluate your system.
    # All you need to be sure to do is create a list of dicts with at least
    # the keys of the dicts returned by `run_gpt` and `run_eleuther`.
    # Add those dicts to `gens`.
    #
    # Here is an example where we just do "Open QA with no context",
    # for an "original system" that would not earn any credit (since
    # it is not original!):
    for question in questions:
        gens[question] = answer_exec(question)

    # Quick tests we advise you to run:
    # 1. Make sure `gens` is a dict with the questions as the keys:
    assert all(q in gens for q in questions)
    # 2. Make sure the values are dicts and have the key we will use:
    assert all(isinstance(d, dict) and "generated_answer" in d for d in gens.values())

    # And finally the output file:
    with open("cs224u-openqa-bakeoff-entry.json", "wt") as f:
        json.dump(gens, f, indent=4)







if __name__ == '__main__':
    # create_bakeoff_submission()
    # print(dev_exs)
    def evaluate_few_shot_open_qa(
            examples,
            batch_size=20,
            gen_func=run_eleuther,
            joiner="\n\n"):
        prompts = []

        gens = []

        for i in range(0, len(examples)):
            question = examples[i].question

            passages, passage_probs = get_passages_with_scores(question)
            train_exs = []
            for squadexample in dev_exs:
                if squadexample.question == question or squadexample.title in question:
                    train_exs.append(squadexample)
            ps = [build_few_shot_open_qa_prompt(question, psg, train_exs) for psg in passages]
            ps2, gs = answer_scoring_ev(passages, passage_probs, ps)
            prompts.append(ps2)
            gens.append(gs)
            print(len(prompts))
            print(len(gens))
        return evaluate(examples, prompts, gens)


    results = evaluate_few_shot_open_qa(dev_exs)
    print(results['macro_f1'])
