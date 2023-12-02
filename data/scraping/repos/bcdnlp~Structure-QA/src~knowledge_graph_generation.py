import json
import logging
import os
import pickle
import re
import time
import yaml
from time import sleep

# constrain the use of gpus
with open('src/config.yml') as f:
    config = yaml.safe_load(f)
os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([str(g) for g in config['gpu_no']])

from tqdm import tqdm

import openai
import torch
import spacy
from transformers import T5Tokenizer, T5ForConditionalGeneration

from utils import clean_string, get_file_path
from src.demonstration import graph_demo_generation

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.FileHandler(os.path.join('log/', 'ckg.log'), 'w')
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def flatten(contexts):
    index = []
    flatten_contexts = []
    for idx, context in enumerate(contexts):
        passages = context.split('\n\n')
        for passage in passages:
            index.append(idx)
            flatten_contexts.append(passage)
    return index, flatten_contexts

def unflatten(index, flatten_results):
    results = []
    prev_idx = -1
    for idx, result in zip(index, flatten_results):
        if idx != prev_idx:
            results.append([])
            prev_idx = idx
        results[-1].append(result)

    for i in range(len(results)):
        results[i] = '\n\n'.join(results[i])

    return results

def spacy_ner(contexts):
    """
        build based on https://aclanthology.org/N19-1240.pdf
    """
    nlp = spacy.load("en_core_web_trf")

    index, flatten_contexts = flatten(contexts)
    entities = []
    for doc in tqdm(nlp.pipe(flatten_contexts), desc='spacy'):
        entities_text = [ent.text.strip("().'") for ent in doc.ents]
        entities_text = '\n'.join(entities_text)
        entities.append(entities_text)
    entities = unflatten(index, entities)

    return entities

def manual_ner(contexts):
    path = config['_'.join([config['dataset_name'], 'path'])]
    with open(path, 'r') as f:
        all_data = json.load(f)

    entities = []
    for data in all_data:
        entities.append(data['entities'])

    return entities

def llm_ner(contexts):
    index, flatten_contexts = flatten(contexts)
    prompts = []

    entity_prompt = 'Entities:\n'
    for context in tqdm(flatten_contexts):
        # real question
        context = '\n'.join(['Document:', context])
        prompt = '\n\n'.join([context, entity_prompt])

        # demonstrations
        demo_graphs = graph_demo_generation(prompt, mode='entity')
        prompt = ''.join([demo_graphs, prompt])

        prompts.append(prompt)

    entities = llm_generation(prompts)
    entities = unflatten(index, entities)

    return entities

def openai_generation(passages):
    with open(config['openai_org_id'], 'r') as f:
        openai.organization = f.readline().strip()

    with open(config['openai_key'], 'r') as f:
        key = json.load(f)
        openai.api_key = key[config['account']].strip()

    predictions = []
    for idx, passage in tqdm(enumerate(passages), desc='Openai generation'):
        message = {'role': 'user', 'content': passage}
        while True:
            try:
                response = openai.ChatCompletion.create(
                                model='gpt-3.5-turbo-0301',
                                messages=[message],
                                temperature=0,
                                max_tokens=config['max_graph_length'],
                                frequency_penalty=0,
                                presence_penalty=0,
                                n=1,
                                )
                break
            except:
                time.sleep(3)
                print("Errrrrrrrrrrrrrrrrrrrrrrrrrrrrrrror")
#                print(response)

        prediction = response['choices'][0]['message']['content']
        predictions.append(prediction)

    return predictions

def t5_generation(passages):
    cuda = torch.device('cuda')
    tokenizer = T5Tokenizer.from_pretrained(config['rel_model'],
                                            model_max_length=config['model_max_length'])
    model = T5ForConditionalGeneration.from_pretrained(config['rel_model'])
    model.eval()
    model.to(device=cuda)

    outputs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(passages), config['graph_batch_size']),
                      desc='LLM generation'):
            batch_passages = passages[i: i + config['graph_batch_size']]
            batch_inputs = tokenizer(batch_passages,
                                     padding='longest',
                                     return_tensors='pt')
            batch_input_ids = batch_inputs.input_ids
            batch_input_ids = batch_input_ids.to(device=cuda)
            batch_attn_masks = batch_inputs.attention_mask
            batch_attn_masks = batch_attn_masks.to(device=cuda)
        
            output_sequences = model.generate(input_ids=batch_input_ids,
                                              attention_mask=batch_attn_masks,
                                              max_new_tokens=50,
                                              do_sample=False)
        
            output = tokenizer.batch_decode(output_sequences,
                                            skip_special_tokens=True)
            outputs.extend(output)

    return outputs

def llm_generation(passages):
    if 't5' in config['rel_model']:
        return t5_generation(passages)
    elif 'openai' == config['rel_model']:
        return openai_generation(passages)

def build_graph_by_llm(contexts, entities, questions):
#    relation_prompt = "Give me relations between named entities in the document in (entity, relation, entity) format:"
#    relation_prompt = 'Given the above document, please output shorten relations between named entities in (entity, relation, entity) format:'
    relation_prompt = 'Graph:\n'

    # relation generation and graph building
    index, flatten_entities = flatten(entities)
    if 'none' == config['relation']:
        logger.info('Generating entity graphs')
        graphs = []
        for entities in flatten_entities:
            entities = entities.split('\n')
            entities = ['"' + entity + '"' if ',' in entity else entity
                        for entity in entities]
            entities = ', '.join(entities)
            graph = ''.join(['(', entities, ')'])
            graphs.append(graph)
    else:
        logger.info('Generating graphs with relations')
        _, flatten_contexts = flatten(contexts)
        flatten_questions = []
        prev_idx = -1
        question_idx = -1
        for idx, _ in zip(index, flatten_contexts):
            if idx != prev_idx:
                question_idx += 1
                prev_idx = idx
            flatten_questions.append(questions[question_idx])

        prompts = []
        for context, entities, question in zip(flatten_contexts,
                                               flatten_entities,
                                               flatten_questions):
            context = '\n'.join(['Document:', context])
            entities = '\n'.join(['Entities:', entities])
            prompt = '\n\n'.join([context, entities])
            if 'useful' == config['relation']:
                question = '\n'.join(['Question:', question])
                prompt = '\n\n'.join([prompt, question])
            prompt = '\n\n'.join([prompt, relation_prompt])
            if 'few-shot' == config['rel_shot']:
                demo_graphs = graph_demo_generation(prompt, mode='graph')
                prompt = ''.join([demo_graphs, prompt])
            prompts.append(prompt)

        graphs = llm_generation(prompts)

    graphs = unflatten(index, graphs)
    for i in range(len(graphs)):
        graphs[i] = re.sub('\n\n', '\n', graphs[i])

    # check outputs format
    logger.info(graphs[0])
#    input()

    return graphs

def construct_entity_graph(context_qa_tuples_text):
    logger.info('Start graph building')

    # return existing context_qa_tuples
    file_path = get_file_path(config['context_graph_qa_tuple_text_pickle'],
                              category='fetching_results')
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as f:
            context_graph_qa_tuples_text = pickle.load(f)

        logger.info('Load built graph results from ' + file_path)

        return context_graph_qa_tuples_text

    idxes, contexts, questions, answers = list(zip(*context_qa_tuples_text))
    # split context into paragraphs content
    processed_contexts = []
    for context in contexts:
        context = context.split('\n')
        context = [context[i] for i in range(1, len(context), 2)]
        context = '\n\n'.join(context)
        processed_contexts.append(context)

    logger.info('Entity extraction method: %s'%(config['ner_model']))

    # ner: spacy
    ner_path = get_file_path(config['context_entity_qa_tuple_text_pickle'],
                             category='fetching_results')
    logger.info(ner_path)
    if os.path.isfile(ner_path):
        with open(ner_path, 'rb') as f:
            context_entity_qa_tuples_text = pickle.load(f)
            entities = list(zip(*context_entity_qa_tuples_text))[2]
        logger.info('Load ner results from ' + ner_path)
    else:
        # handle each paragraph individually
        if 'spacy' == config['ner_model']:
            entities = spacy_ner(processed_contexts)
        elif 'manual-ner' == config['ner_model']:
            entities = manual_ner(processed_contexts)
        elif 'openai' == config['ner_model']:
            entities = llm_ner(processed_contexts)

        context_entity_qa_tuples_text = list(zip(idxes, contexts, entities, questions, answers))
        with open(ner_path, 'wb') as f:
            pickle.dump(context_entity_qa_tuples_text, f)

    # relation and graph building: openai gpt
    if 'openai' == config['rel_model'] or 't5' in config['rel_model']:
        graphs = build_graph_by_llm(processed_contexts, entities, questions)
    elif 'manual-rel' == config['rel_model']:
        pass

    context_graph_qa_tuples_text = list(zip(idxes, contexts, graphs, questions, answers))

    logger.info('Saving Results to ' + file_path)
    # save for future use
    with open(file_path, 'wb') as f:
        pickle.dump(context_graph_qa_tuples_text, f)

    logger.info("Finish graph building")

    return context_graph_qa_tuples_text

def construct_entity_graph_directly(context_qa_tuples_text):
    """ for kg2
    """
    logger.info('Start graph building')

    # return existing context_qa_tuples
    file_path = get_file_path(config['context_graph_qa_tuple_text_pickle'],
                              category='fetching_results')
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as f:
            context_graph_qa_tuples_text = pickle.load(f)

        logger.info('Load built graph results from ' + file_path)

        return context_graph_qa_tuples_text

    idxes, contexts, questions, answers = list(zip(*context_qa_tuples_text))

    logger.info('Generating graphs with relations')
    index, flatten_contexts = flatten(contexts)

    relation_prompt = 'Graph:\n'
    prompts = []
    for context in flatten_contexts:
        # real question
        context = '\n'.join(['Document:', context])
        prompt = '\n\n'.join([context, relation_prompt])

        # demonstrations, default: few-shot demos, kg2
        demo_graphs = graph_demo_generation(prompt, mode='graph')
        prompt = ''.join([demo_graphs, prompt])

        prompts.append(prompt)

    graphs = llm_generation(prompts)

    graphs = unflatten(index, graphs)
    for i in range(len(graphs)):
        graphs[i] = re.sub('\n\n', '\n', graphs[i])

    context_graph_qa_tuples_text = list(zip(idxes, contexts, graphs, questions, answers))

    logger.info('Saving Results to ' + file_path)
    # save for future use
    with open(file_path, 'wb') as f:
        pickle.dump(context_graph_qa_tuples_text, f)

    logger.info("Finish graph building")

    return context_graph_qa_tuples_text

def construct_empty_graph(context_qa_tuples_text):
    logger.info('Start graph building')

    # return existing context_qa_tuples
    file_path = get_file_path(config['context_graph_qa_tuple_text_pickle'],
                              category='fetching_results')
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as f:
            context_graph_qa_tuples_text = pickle.load(f)

        logger.info('Load built graph results from ' + file_path)

        return context_graph_qa_tuples_text

    idxes, contexts, questions, answers = list(zip(*context_qa_tuples_text))
    graphs = [''] * len(idxes)
    context_graph_qa_tuples_text = list(zip(idxes, contexts, graphs, questions, answers))

    logger.info('Saving Results to ' + file_path)
    # save for future use
    with open(file_path, 'wb') as f:
        pickle.dump(context_graph_qa_tuples_text, f)

    logger.info("Finish graph building")

    return context_graph_qa_tuples_text

def manual_graph():
    logger.info('Start graph building')

    # return existing context_qa_tuples
    file_path = get_file_path(config['context_graph_qa_tuple_text_pickle'],
                              category='fetching_results')
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as f:
            context_graph_qa_tuples_text = pickle.load(f)

        logger.info('Load built graph results from ' + file_path)

        return context_graph_qa_tuples_text

    path = config['_'.join([config['dataset_name'], 'path'])]
    with open(path, 'r') as f:
        all_data = json.load(f)

    context_graph_qa_tuples_text = []
    for data in all_data:
        tuple_text = [data['_id'],
                      data['documents'],
                      data['graph1'],
                      data['question'],
                      data['answer']]
        if 'full' == config['relation']:
            tuple_text[2] = data['graph2']

        tuple_text = tuple(tuple_text)
        context_graph_qa_tuples_text.append(tuple_text)

    logger.info('Saving Results to ' + file_path)
    # save for future use
    with open(file_path, 'wb') as f:
        pickle.dump(context_graph_qa_tuples_text, f)

    logger.info("Finish graph building")

    return context_graph_qa_tuples_text

def construct_multi_level_graph(context_qa_tuple_text):
    pass

def graph_generation(context_qa_tuples_text):
    if 'none' == config['graph_type']:
        tuple_text = context_qa_tuples_text
    elif 'empty' == config['graph_type']:
        tuple_text = construct_empty_graph(context_qa_tuples_text)
    elif 'manual-entity' == config['graph_type']:
        tuple_text = manual_graph()
    elif 'direct-entity' == config['graph_type']:
        tuple_text = construct_entity_graph_directly(context_qa_tuples_text)
    elif 'entity' == config['graph_type']:
        tuple_text = construct_entity_graph(context_qa_tuples_text)
    elif 'multi_node' == config['graph_type']:
        tuple_text = construct_multi_level_graph(context_qa_tuples_text)

    return tuple_text

if __name__ == '__main__':
    from src.dataset_reader import read_data
    from src.context_fetching import context_fetching
    qa_pairs = read_data(config['dataset_name'])
    context_qa_tuples = context_fetching(qa_pairs)
    tuple_text = graph_generation(context_qa_tuples)
    print(tuple_text[0])

