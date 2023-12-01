from helper_synthesis import get_examples_as_text, xml_to_json, is_valid_xml, check_valid_aspect_xml, count_sentences_in_text, german_language_detected
from IPython.display import clear_output
from dotenv import load_dotenv
from llama_cpp import Llama
import openai
import time
import json
import uuid
import os
import sys  # Import the sys module to access command-line arguments

import time
print(time.strftime("%d.%m.%Y %H:%M:%S"))

############### Settings ###############

load_dotenv()

# Default values
SPLIT = 0
MODEL_ID = 0
FEW_SHOTS = "random"

# Check if command-line arguments are provided
if len(sys.argv) > 1:
    # Parse command-line arguments
    MODEL_ID = int(sys.argv[1])
    SPLIT = int(sys.argv[2])
    FEW_SHOTS = sys.argv[3]

DATASET_PATH = f'../07 train classifier/real/split_{SPLIT}.json'
LABELS_AND_EXAMPLES_PATH = f"few_shot_examples/few_shot_examples_{FEW_SHOTS}.json"

# LLM Settings
MAX_TOKENS = 512
CONTEXT_SIZE = 4096
TEMPERATURE = 1.0

# Set Seed
SEED = int(str(43) + str(SPLIT) + str(MODEL_ID))

N_RETRIES = 25

# Setup Classes/Polarities for Synthesis
CLASSES = ["GENERAL-IMPRESSION", "FOOD", "SERVICE", "AMBIENCE", "PRICE"]
POLARITIES = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
COMBINATIONS = [(aspect, polarity)
                for polarity in POLARITIES for aspect in CLASSES]

STOP_CRITERIA = ["\n"]

MODELS = ["Llama13B", "Llama70B", "Falcon40B", "GPT-3"]
# 175B, 70B und 40B
MODEL_PATHS = {"Llama13B": "llm_models/llama-2-13b.Q4_0.gguf",
               "Llama70B": "llm_models/llama-2-70b.Q4_0.gguf", "Falcon40B": "llm_models/falcon-40b-Q4_K_S.gguf"}
MODEL_NAME = MODELS[MODEL_ID]

SYNTH_PATH = f"../07 train classifier/synth/{MODEL_NAME}/{FEW_SHOTS}/split_{SPLIT}.json"


# Status

print("Split:", SPLIT, "Model:", MODELS[MODEL_ID], "Few-Shot Setting:", FEW_SHOTS)

############### Code ###############

with open('../prompt_template.txt', 'r') as file:
    PROMPT_TEMPLATE = file.read()

with open(DATASET_PATH, 'r', encoding='utf-8') as json_file:
    dataset = json.load(json_file)

# Setup Model

if MODEL_NAME == "Llama70B":
    llm = Llama(model_path=MODEL_PATHS[MODEL_NAME], seed=SEED,
                n_gpu_layers=44, n_ctx=CONTEXT_SIZE, verbose=False, n_gqa=8)
    clear_output(wait=False)

    def llm_model(text):
        return llm(text, max_tokens=MAX_TOKENS, stop=STOP_CRITERIA, echo=True, top_p=1, temperature=TEMPERATURE)["choices"][0]["text"][len(text):]

if MODEL_NAME == "Llama13B" or MODEL_NAME == "Falcon40B":
    llm = Llama(model_path=MODEL_PATHS[MODEL_NAME], seed=SEED,
                n_gpu_layers=1, n_ctx=CONTEXT_SIZE, verbose=False)
    clear_output(wait=False)

    def llm_model(text):
        return llm(text, max_tokens=MAX_TOKENS, stop=STOP_CRITERIA, echo=True, top_p=1, temperature=TEMPERATURE)["choices"][0]["text"][len(text):]

if MODEL_NAME == "GPT-3":
    openai.api_key = os.getenv("OPENAI_API_KEY")

    def llm_model(text):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": text}
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            stop=STOP_CRITERIA
        )
        return response.choices[0].message.content.strip()


# Load Labels and Examples

with open(LABELS_AND_EXAMPLES_PATH, 'r', encoding='utf-8') as json_file:
    labels_and_examples = json.load(json_file)[f"split_{SPLIT}"]

labels = labels_and_examples["labels_for_prediction"]
labels = [[(aspect, polarity) for aspect, polarity in sub_list]
          for sub_list in labels]

examples = labels_and_examples["few_shot_ids"]

# Create Synthetic Examples

synth_dataset = []

for idx, label in enumerate(labels):
    # Setup Statistics
    invalid_xml_schema = 0
    invalid_xml_tags = 0
    aspect_polarity_in_text_but_not_in_label = 0
    more_than_one_sentences = 0
    no_german_language = 0

    # Setup JSON for new synth example
    synth_example = {}
    synth_example["llm_retry_statistic"] = []

    found_valid_example = False

    # Alle 25 Beispielsets sollen 25 mal versucht werden
    for new_example_idx in range(len(examples[str(idx)])):
        for retry in range(N_RETRIES):
            # new_example_idx will change in case it wasn't possible to generate a text for a given label after N_MAX_NEW_EXAMPLES retires
            few_shot_examples = examples[str(idx)][f"{new_example_idx}"]
            few_shot_examples = [
                entry for entry in dataset if entry['id'] in few_shot_examples]

            # Build Prompt
            examples_text = get_examples_as_text(few_shot_examples)
            prompt_footer = f'\nLabel:{str(label)}\nPrediction:'
            prompt = PROMPT_TEMPLATE + examples_text + prompt_footer

            # Execute LLM
            start_time_prediction = time.time()
            prediction = llm_model(prompt).lstrip()
            end_time_prediction = time.time()
            prediction_duration = end_time_prediction - start_time_prediction
            print("prediction took:", prediction_duration)

            if is_valid_xml(f'<input>{prediction}</input>') == False:
                invalid_xml_schema += 1
            else:
                if check_valid_aspect_xml(f'<input>{prediction}</input>') == False:
                    invalid_xml_tags += 1
                else:
                    prediction_as_json = xml_to_json(
                        prediction, label, MODEL_NAME, SPLIT)
                    if prediction_as_json == "not-in-label":
                        aspect_polarity_in_text_but_not_in_label += 1
                    else:
                        if count_sentences_in_text(prediction_as_json["text"]) > 1:
                            more_than_one_sentences += 1
                        else:
                            if german_language_detected(prediction_as_json["text"]) == False:
                                no_german_language += 1
                            else:
                                synth_example["id"] = str(uuid.uuid4())
                                synth_example["llm_label"] = label
                                synth_example["llm_examples"] = few_shot_examples
                                synth_example["llm_prompt"] = prompt
                                synth_example["llm_prediction_raw"] = prediction
                                synth_example["llm_invalid_xml_schema"] = invalid_xml_schema
                                synth_example["llm_invalid_xml_tags"] = invalid_xml_tags
                                synth_example["llm_aspect_polarity_in_text_but_not_in_label"] = aspect_polarity_in_text_but_not_in_label
                                synth_example["llm_more_than_one_sentences"] = more_than_one_sentences
                                synth_example["llm_no_german_language"] = no_german_language
                                synth_example["llm_prediction_duration"] = prediction_duration
                                for key in prediction_as_json.keys():
                                    synth_example[key] = prediction_as_json[key]
                                found_valid_example = True

            # Log current generation
            print(
                f'current index: {idx}, n_retry: {len(synth_example["llm_retry_statistic"])}, text: {prediction}')

            if found_valid_example:
                break
            else:
                # Save Statistics of retries
                retry_statistic = {}
                retry_statistic["llm_label"] = label
                retry_statistic["llm_examples"] = [example["id"] for example in few_shot_examples]
                retry_statistic["llm_prompt"] = prompt
                retry_statistic["llm_prediction_raw"] = prediction
                retry_statistic["llm_invalid_xml_schema"] = invalid_xml_schema
                retry_statistic["llm_invalid_xml_tags"] = invalid_xml_tags
                retry_statistic["llm_aspect_polarity_in_text_but_not_in_label"] = aspect_polarity_in_text_but_not_in_label
                retry_statistic["llm_more_than_one_sentences"] = more_than_one_sentences
                retry_statistic["llm_no_german_language"] = no_german_language
                retry_statistic["llm_change_examples"] = new_example_idx
                retry_statistic["llm_retries_for_example_set"] = retry
                retry_statistic["llm_prediction_duration"] = prediction_duration
                synth_example["llm_retry_statistic"].append(retry_statistic)

        if found_valid_example:
            synth_dataset.append(synth_example)
            break

os.makedirs(os.path.dirname(SYNTH_PATH), exist_ok=True)
with open(SYNTH_PATH, "w") as outfile:
    json.dump(synth_dataset, outfile)


print(time.strftime("%d.%m.%Y %H:%M:%S"))