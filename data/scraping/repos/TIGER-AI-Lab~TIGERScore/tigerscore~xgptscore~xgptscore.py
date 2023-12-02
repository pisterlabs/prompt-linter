import os
import json
import logging
from .process import MODE_PROCESS_MAP
from .process_utils import XPGTItem, truncate_items, get_query_messages
from .openai_utils import openai_completions, _chatml_to_prompt
from typing import List, Union
from dacite import from_dict
from pathlib import Path
from functools import partial


def xgptscore(
    items: List[Union[XPGTItem, dict]],
    mode: str,
    model_name: str,
    num_workers: int = None,
    batch_size: int = None,
    **kwargs,
):
    config_path = os.path.join(os.path.dirname(
        __file__), f"mode_configs/{mode}.json")
    config_path = Path(config_path)
    if not config_path.exists():
        logging.warning(
            f"Config file {config_path} does not exist. Use default config.")
        config_path = config_path.with_name("default.json")

    with open(config_path, "r") as f:
        config = json.load(f)
    config.update(kwargs)
    if "max_lengths" in config:
        items = truncate_items(items, config["max_lengths"])

    if isinstance(items[0], dict):
        items = [from_dict(data_class=XPGTItem, data=item) for item in items]
    process_func = MODE_PROCESS_MAP[mode]
    if "process_kwargs" in config:
        process_func = partial(process_func, **config["process_kwargs"])
    process_results = list(map(process_func, items))

    total_round = len([x for x in process_results[0] if x['do_query']])
    logging.warning(f"Total chat rounds: {total_round}")
    logging.warning(f"Total chat messages: {len(items)}")
    # query and process
    round = 0
    queried_messages = [[] for _ in range(len(items))]
    total_price = 0
    total_time = 0
    round_completions = []
    while True:
        round += 1
        logging.warning(f"Processing chat round {round}/{total_round}")
        query_messages = list(
            map(get_query_messages, process_results, queried_messages))
        query_messages, postprocess_funcs = list(zip(*query_messages))
        chatml_prompts = list(map(_chatml_to_prompt, query_messages))
        openai_results = openai_completions(
            chatml_prompts,
            model_name=model_name,
            num_procs=num_workers,
            batch_size=batch_size,
            **config['decoding'],
        )
        completions = openai_results['completions']
        total_price += sum(openai_results['price_per_example'])
        total_time += sum(openai_results['time_per_example'])
        logging.warning(f"Round {round} price: {total_price}$")
        logging.warning(f"Round {round} time: {total_time}")
        postprocess_completions = [postprocess_funcs[idx](
            completion) for idx, completion in enumerate(completions)]
        round_completions.append(postprocess_completions)
        for idx, completion in enumerate(completions):
            queried_messages[idx] = query_messages[idx] + \
                [{"role": "assistant", "content": completion}
                 ]  # add the assistant response
        if round == total_round:
            _query_messages = list(
                map(get_query_messages, process_results, queried_messages))
            assert all([x is None for x in _query_messages]
                       ), "All messages should be queried"
            break
    logging.warning(f"Total price: {total_price}$")
    logging.warning(f"Total time: {total_time}")
    logging.warning(f"Total time per example: {total_time / len(items)}")
    round_completions = list(zip(*round_completions))
    return dict(
        round_completions=round_completions,
        messages_records=queried_messages,
    )


"""
Example Usage:
task = "translation"
with open("example.json", "r") as f:
    items = json.load(f)
xgptitems = []
for item in items:
    for cand in item['candidates']:
        xgptitems.append(XPGTItem(
            task=task,
            instruction=item['instruction'],
            input=item['input'],
            ref_output=item['output'],
            hypo_output=cand['text']
        ))
result = xgptscore(xgptitems, "ea", "ChatGPT")
idx = 0
for item in items:
    for cand in item['candidates']:
        cand['responses'] = result['round_completions'][idx]
        cand['messages_records'] = result['messages_records'][idx]
json.dump(items, open("example_result.json", "w"), indent=4, ensure_ascii=False)
"""
