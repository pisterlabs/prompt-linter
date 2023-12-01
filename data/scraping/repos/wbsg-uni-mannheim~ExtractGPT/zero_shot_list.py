import json
import random
from datetime import datetime
from json import JSONDecodeError

import click
import torch
from dotenv import load_dotenv
from langchain import LLMChain, HuggingFacePipeline
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import SystemMessage
from pydantic import ValidationError
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from pieutils import create_pydanctic_models_from_known_attributes, save_populated_task, parse_llm_response_to_json
from pieutils.evaluation import evaluate_predictions
from pieutils.fusion import fuse_models
from pieutils.preprocessing import update_task_dict_from_test_set


@click.command()
@click.option('--dataset', default='oa-mine', help='Dataset Name')
@click.option('--model', default='gpt-3.5-turbo-0613', help='Model name')
@click.option('--verbose', default=True, help='Verbose mode')
def main(dataset, model, verbose):
    # Load task template
    with open('prompts/task_template.json', 'r') as f:
        task_dict = json.load(f)

    task_dict['task_name'] = "multiple_attribute_values-great"
    task_dict['task_prefix'] = "Extract the attribute values from the product {part} in a JSON format. Valid " \
                               "attributes are {attributes}. If an attribute is not present in the product title, " \
                               "the attribute value is supposed to be 'n/a'."

    # task_dict['data_source'] = dataset
    task_dict['dataset_name'] = dataset
    task_dict['timestamp'] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Load examples into task dict
    task_dict = update_task_dict_from_test_set(task_dict)

    # Initialize model
    task_dict['model'] = model
    if 'gpt-3' in task_dict['model'] or 'gpt-4' in task_dict['model']:
        llm = ChatOpenAI(model_name=task_dict['model'], temperature=0)
    else:
        if 'StableBeluga' in task_dict['model']:
            tokenizer = AutoTokenizer.from_pretrained(task_dict['model'], use_fast=False,
                                                      cache_dir="/ceph/alebrink/cache")
            model = AutoModelForCausalLM.from_pretrained(task_dict['model'], torch_dtype=torch.float16,
                                                         low_cpu_mem_usage=True, device_map="auto")
            model.tie_weights()

            pipe = pipeline(
                "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256, temperature=0, top_p=0.95)
            llm = HuggingFacePipeline(pipeline=pipe)
        elif 'SOLAR' in task_dict['model']:
            tokenizer = AutoTokenizer.from_pretrained(task_dict['model'], cache_dir="/ceph/alebrink/cache")
            model = AutoModelForCausalLM.from_pretrained(
                task_dict['model'],
                device_map="auto",
                torch_dtype=torch.float16,
                load_in_8bit=True,
                rope_scaling={"type": "dynamic", "factor": 2}  # allows handling of longer inputs
            )
            model.tie_weights()
            pipe = pipeline(
                "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256, temperature=0, use_cache=True)
            llm = HuggingFacePipeline(pipeline=pipe)
        else:
            print('Unknown model!')

    # Save populated task and make sure that it is saved in the correct folder!
    save_populated_task(task_dict['task_name'], task_dict)

    pydantic_models = create_pydanctic_models_from_known_attributes(task_dict['known_attributes'])
    # Create chains to extract attribute values from product titles
    system_setting = SystemMessage(
        content='You are a world-class algorithm for extracting information in structured formats. ')
    human_task_prompt = HumanMessagePromptTemplate.from_template(task_dict['task_prefix'])
    human_message_prompt = HumanMessagePromptTemplate.from_template("{input}")
    prompt = ChatPromptTemplate(messages=[system_setting, human_task_prompt, human_message_prompt])

    chain = LLMChain(
        prompt=prompt,
        llm=llm,
        verbose=verbose
    )

    def select_and_run_llm(category, input_text, part='title'):
        pred = None
        if len(input_text) == 0:
            # No input text provided.
            return pred
        response = chain.run(input=input_text, attributes=', '.join(task_dict['known_attributes'][category]), part=part)
        try:
            # Convert json string into pydantic model
            pred = pydantic_models[category](**json.loads(response))
            if verbose:
                json_response = json.loads(response)
                print(json_response)
        except ValidationError as valError:
            print(valError)
        except JSONDecodeError as e:
            converted_response = parse_llm_response_to_json(response)
            try:
                pred = pydantic_models[category](**converted_response)
            except ValidationError as valError:
                print(valError)
            except Exception as e:
                print(e)
        except Exception as e:
            print(e)
            print('Response: ')
            print(response)
        return pred

    with get_openai_callback() as cb:
        # Mave dataset has multiple sources. We need to iterate over all of them.
        if dataset == 'mave':
            part_predictions = {}
            parts = ['title', 'description', 'features', 'attributes']
            # Run LLM for each part of the product
            for part in parts:
                example_part = part if part != 'title' else 'input'
                example_part = example_part if example_part != 'attributes' else 'rest'
                preds = [select_and_run_llm(example['category'], example[example_part], part=part) for example in
                         tqdm(task_dict['examples'], disable=not verbose)]
                part_predictions[example_part] = preds

            # Combine all predictions using a majority vote
            preds = [fuse_models(pydantic_models[example['category']], pred_title, pred_description, pred_features,
                                 pred_rest)
                     for example, pred_title, pred_description, pred_features, pred_rest
                     in zip(task_dict['examples'], part_predictions['input'], part_predictions['description'],
                            part_predictions['features'], part_predictions['rest'])]
        elif dataset == 'mave_v2':
            hashed_predictions = {category: {} for category in
                                  task_dict['known_attributes']}  # Cache predictions per category
            preds = []
            # Run LLM for each part of the product
            for example in tqdm(task_dict['examples'], disable=not verbose):
                example_preds = []
                hashed_input = hash(example['input'])
                if hashed_input in hashed_predictions[example['category']]:
                    example_preds.append(hashed_predictions[example['category']][hashed_input])
                    print("Cached")
                    continue
                title_prediction = select_and_run_llm(example['category'], example['input'], part='title')
                hashed_predictions[example['category']][hashed_input] = title_prediction
                example_preds.append(title_prediction)

                for description in example['paragraphs']:
                    hashed_description = hash(description)
                    if hashed_description in hashed_predictions[example['category']]:
                        example_preds.append(hashed_predictions[example['category']][hashed_description])
                        print("Cached")
                        continue

                    description_prediction = select_and_run_llm(example['category'], description, part='description')
                    hashed_predictions[example['category']][hashed_description] = description_prediction
                    example_preds.append(description_prediction)

                pred = fuse_models(pydantic_models[example['category']], *example_preds)
                if verbose:
                    print(pred)
                preds.append(pred)
        else:
            preds = [select_and_run_llm(example['category'], example['input']) for example in
                     tqdm(task_dict['examples'], disable=not verbose)]

        task_dict['total_tokens'] = cb.total_tokens
        print(f"Total Tokens: {cb.total_tokens}")

    # Calculate recall, precision and f1
    task_dict['results'] = evaluate_predictions(preds, task_dict)
    # Save populated task and make sure that it is saved in the correct folder!
    save_populated_task(task_dict['task_name'], task_dict)


if __name__ == '__main__':
    load_dotenv()
    random.seed(42)
    main()
