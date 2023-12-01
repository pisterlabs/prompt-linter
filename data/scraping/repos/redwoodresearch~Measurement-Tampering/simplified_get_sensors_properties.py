import json
import os
import random
import threading
from collections import defaultdict

import attrs
import openai
import tiktoken
from cattrs.preconf.json import make_converter

from text_properties.input_text_properties import QueryItems, get_completions, make_query
from text_properties.simplified import sensors_simple
from text_properties.simplified_data_types import SimpleFullDatum, SimpleWritingResponse

# %%

tokenizer = tiktoken.encoding_for_model("gpt-4")

if "OPENAI_API_KEY" not in os.environ:
    openai.api_key_path = os.path.expanduser("~/.openai_api_key")

json_converter = make_converter()

# %%

# with open("writing_responses_all_val_extra_new.jsonl", "r") as f:
with open("writing_responses_out_v2_more_tamp.jsonl", "r") as f:
    writing_responses = [json_converter.loads(line, SimpleWritingResponse) for line in f.readlines()]

# %%

text_to_writing_response = {x.cut_text: x for x in writing_responses}

# %%

# writing_responses_4 = [writ for writ in writing_responses if writ.model == "gpt-4-0613"]

# %%

simple_sensor_query_items: QueryItems = []

for resp in writing_responses:
    simple_sensor_query_items.append(
        attrs.evolve(
            make_query(
                resp.final_text,
                sensors_simple,
                resp.full_response.parsed_response.query.full_text,
            ),
            cut_text=resp.full_response.parsed_response.query.cut_text,
        )
    )


len(simple_sensor_query_items)

# %%

# out_file = "sensors_out.jsonl"
out_file = "sensors_out_more_tamp_new_approach.jsonl"

with open(out_file, "w") as f:
    ...

# TODO: fix threads
n_threads = max(min(len(simple_sensor_query_items), 5), 1)

shuffle_items = list(simple_sensor_query_items)
random.shuffle(shuffle_items)

threads = [
    threading.Thread(
        target=get_completions,
        args=(
            shuffle_items,
            k,
            n_threads,
            ["gpt-4-0613"],  # for whatever reason 3.5 turbo just can't do this??? (whatever I guess...)
            # ["gpt-3.5-turbo-16k"],
            out_file,
            50,
            # dict(
            #     functions=[
            #         {
            #             "name": "provide_answer",
            #             "description": "Answers the question with Yes/No",
            #             "parameters": {
            #                 "type": "object",
            #                 "properties": {
            #                     "answer": {
            #                         "type": "string",
            #                         "description": "The Yes/No answer",
            #                         "enum": ["Yes", "No"],
            #                     },
            #                 },
            #             },
            #             "required": ["answer"],
            #         }
            #     ],
            #     function_call={"name": "provide_answer"},
            # ),
        ),
    )
    for k in range(n_threads)
]

# %%

for t in threads:
    t.start()

# %%

for t in threads:
    t.join()

# %%

print("done!")

# %%

# with open("sensors_out_4_simp_check.jsonl", "r") as f:
with open("sensors_out_more_tamp_new_approach.jsonl", "r") as f:
    # with open("sensors_out_3_5.jsonl", "r") as f:
    loaded_items_sensors = [json.loads(l) for l in f.read().splitlines()]
len(loaded_items_sensors)

# %%


# %%

text_to_sensors: defaultdict[str, dict[str, bool]] = defaultdict(dict)

for x in loaded_items_sensors:
    assert len(x["response"]) == 1
    resp = list(x["response"].values())[0]

    # val = json.loads(resp["choices"][0]["message"]["function_call"]["arguments"])["answer"] == "Yes"

    response_lines = resp["choices"][0]["message"]["content"].splitlines()
    if len(response_lines) != len(sensors_simple):
        continue

    if not all(prop in y for prop, y in zip(sensors_simple, response_lines)):
        continue
    if not all(("Yes" in y) != ("No" in y or "N/A" in y or "Neutral" in y) for y in response_lines):
        continue

    text_to_sensors[x["query"]["cut_text"]] = {sensor: "Yes" in l for sensor, l in zip(sensors_simple, response_lines)}

full_datums = [
    SimpleFullDatum(text_to_writing_response[txt], sensor_vals)
    for txt, sensor_vals in text_to_sensors.items()
    if len(sensor_vals) == len(sensors_simple)
]
len(full_datums)

# %%


with open("full_datum_more_tamp_new_approach.jsonl", "w") as f:
    for x in full_datums:
        f.write(json_converter.dumps(x) + "\n")


# %%

# all_vals = {k : (text_to_sensors_old[k], v) for k, v in text_to_sensors.items()}

# for v in all_vals.values():
#     print(v)


# text_to_sensors_old = text_to_sensors


# %%

actually_positive = [x for x in full_datums if x.has_latent()]
actually_positive_sensors = [x.sensor_values for x in actually_positive]
all_positive_count = sum(all(x.values()) for x in actually_positive_sensors)
len(actually_positive), all_positive_count, all_positive_count / len(actually_positive)

# %%

# %%

with open("hi.txt", "w") as f:
    f.write(actually_positive[24].writing_response.final_text)


# %%

actually_negative = [x for x in full_datums if not x.has_latent()]
actually_negative_sensors = [x.sensor_values for x in actually_negative]
all_positive_tamper_count = sum(all(x.values()) for x in actually_negative_sensors)
len(actually_negative), all_positive_tamper_count, all_positive_tamper_count / len(actually_negative)

# %%

intended_tamper = [x for x in full_datums if x.theoretical_tamper() and not x.has_latent()]
intended_tamper_sensors = [x.sensor_values for x in intended_tamper]
all_positive_tamper_count = sum(all(x.values()) for x in intended_tamper_sensors)
len(intended_tamper), all_positive_tamper_count, all_positive_tamper_count / len(intended_tamper)


# %%

one_or_more_tamp_count = sum(any(x.values()) for x in actually_negative_sensors)

len(actually_negative), one_or_more_tamp_count, one_or_more_tamp_count / len(actually_negative)

# %%
