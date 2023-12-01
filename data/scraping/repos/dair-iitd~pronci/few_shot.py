import torch
torch.set_grad_enabled(False)

import random
import sys
import ipdb
import os
import openai
from tqdm import tqdm
import json
import numpy as np
import math
import pickle

import transformers
# from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
# from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import pipeline
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration, T5EncoderModel
from datasets import Dataset, load_metric

import run_summarization

data = sys.argv[1]
split = sys.argv[2]
filter_ = sys.argv[3]
train_k_nn = int(sys.argv[4])
if sys.argv[5] == 'True':
    static = True
else:
    static = False


if data == 'semeval':
    train_fp = 'data/train_semeval_1int.jsonl'
    test_fp = 'data/test_semeval_1int.jsonl'
    train_k_nn = 0
    fact_k_nn = 0
    beam_size = 5
    num_return_sequences = 1
elif data == 'pnp':
    if split == 'reg':
        if filter_ == 'all':
            train_fp = 'data/train_rand.jsonl'
            dev_fp = 'data/dev_rand.jsonl'
            test_fp = 'data/test_rand.jsonl'
        elif filter_ == 'fil':
            train_fp = 'data/train_rand_filter.jsonl'
            dev_fp = 'data/dev_rand_filter.jsonl'
            test_fp = 'data/test_rand_filter.jsonl'
    elif split == 'nns':
        if filter_ == 'all':
            train_fp = 'data/train.jsonl'
            dev_fp = 'data/dev.jsonl'
            test_fp = 'data/test.jsonl'
        elif filter_ == 'fil':
            train_fp = 'data/train_filter.jsonl'
            dev_fp = 'data/dev_filter.jsonl'
            test_fp = 'data/test_filter.jsonl'

    # train_k_nn = 5
    fact_k_nn = 0
    beam_size = 5
    num_return_sequences = 1

clear_train_cache = True
# static = False
seed = random.random()
metric_type = 'nli' # sacrebleu/bleurt/nli

local_files_only = True
model = 't5-base'
mode = 't5' 

def get_dataset(path, mode=None):
    sentences, targets = [], []
    targets_dict = {}
    for i, line in enumerate(open(path)):
        line = line.strip()
        jline = json.loads(line)
        # if data == 'semeval':
        #     sentences.append(jline['noun_phrase'])
        #     targets.append(' , '.join(jline['interpretations'][:3]))
        # else:
        if not jline['explicit_relation']: 
            jline['explicit_relation'] = jline['nnp']+' '+jline['nn']+' is None of '+jline['nnp']
        input_ = jline['nnp']+' '+jline['nn']        
        if input_ not in targets_dict:
            sentences.append(input_)
            targets_dict[input_] = []
        targets_dict[input_].append(jline['explicit_relation'])
    targets = [' , '.join(targets_dict[key][:3]) for key in sentences]
    dataset = Dataset.from_dict({'inputs': sentences, 'outputs': targets})
    return dataset

train_dataset = get_dataset(train_fp)
print('Train dataset loaded...')
test_dataset = get_dataset(test_fp, mode='test')
print('Test dataset loaded...')

train_index_path = 'models/embeddings/train_index.faiss'
encoder = T5EncoderModel.from_pretrained('t5-base', local_files_only=local_files_only).to('cuda')
tokenizer = T5Tokenizer.from_pretrained('t5-base', local_files_only=local_files_only)
if not os.path.exists(train_index_path) or clear_train_cache:
    # ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to('cuda')
    # ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    # train_with_embeddings = train_dataset.map(lambda example: {'embeddings': ctx_encoder(**ctx_tokenizer(example["inputs"], return_tensors="pt").to('cuda'))[0][0].cpu().numpy()})
    train_with_embeddings = train_dataset.map(lambda example: {'embeddings': \
        encoder(**tokenizer(example["inputs"], return_tensors="pt").to('cuda')).last_hidden_state[0].max(dim=0)[0].cpu().numpy()})
    train_with_embeddings.add_faiss_index(column='embeddings')
    train_with_embeddings.save_faiss_index('embeddings', train_index_path)
    train_dataset = train_with_embeddings
else:
    train_dataset.load_faiss_index('embeddings', train_index_path)

# if fact_k_nn != 0:
#     facts_trainD = pickle.load(open(facts_train_fp,'rb'))
#     facts_testD = pickle.load(open(facts_test_fp,'rb'))

if mode == 'read':
    predictions = [l.strip() for l in open(ofp,'r').readlines()]
else:
    # none_predictions = open(none_fp,'r').readlines()
    # none_predictions = [int(n.strip()) for n in none_predictions]
    # print(f'Number of none predictions read from {none_fp} = ', len(none_predictions)-sum(none_predictions))

    print('Creating few shot inputs...')
    if static:
        random.seed(seed)
        range_dataset = list(range(len(train_dataset)))
        random.shuffle(range_dataset)
        random_k_examples = range_dataset[:train_k_nn]
        random_train_outputs = [train_dataset[k]["outputs"] for k in random_k_examples]
        random_train_inputs = [train_dataset[k]["inputs"] for k in random_k_examples]

    prompts, input_nps = [], []
    for example_i, example in enumerate(tqdm(test_dataset)):
        # if example_i > 100:
        #     break
        test_input, target = example['inputs'], example['outputs']
        _1_noun, _2_noun = test_input.split()
        # if data != 'semeval' and none_predictions[example_i] == 0:
        #     prompt = 'NONE'
        # else:
        # few_shot_input = 'Compound Noun Paraphrasing\n\n'
        if not static:
            # test_embedding = q_encoder(**q_tokenizer(test_input, return_tensors="pt").to('cuda'))[0][0].numpy()
            test_embedding = encoder(**tokenizer(test_input, return_tensors="pt").to('cuda')).last_hidden_state[0].max(dim=0)[0].cpu().numpy()
            if train_k_nn > 0:
                train_scores, train_retrieved_examples = train_dataset.get_nearest_examples('embeddings', test_embedding, k=train_k_nn)
                train_outputs = train_retrieved_examples["outputs"][::-1]
                train_inputs = train_retrieved_examples["inputs"][::-1]
            else:
                train_outputs, train_inputs = [], []

            # if fact_k_nn != 0:
            #     train_fact_descs = [facts_trainD[train_input] for train_input in train_inputs]
            #     train_fact_descs = ['<fact> '+' , '.join([r.replace(';','') for r in tfd[:fact_k_nn]])+' </fact>' for tfd in train_fact_descs]
                
            # fact_scores, fact_retrieved_examples = facts_dataset.get_nearest_examples('embeddings', test_embedding, k=fact_k_nn)
            # fact_outputs = '<fact> '+' , '.join([r.replace(';','') for r in fact_retrieved_examples["facts"]])+' </fact>'
        else:
            train_outputs = random_train_outputs
            train_inputs = random_train_inputs

        if data == 'semeval':
            prompt = ' , '.join(train_outputs)+f' {test_input} is a {_2_noun} <extra_id_0> {_1_noun}'# , {test_input} is a {_2_noun} <extra_id_1> {_1_noun} '
        else:
            if train_k_nn != 0:
                prompt = ' . '.join(train_outputs)+f' . {test_input} <extra_id_0> {_1_noun} <extra_id_1>'
            else:
                prompt = f'{test_input} is a <extra_id_0> the {_1_noun} <extra_id_1>'

        prompts.append(prompt)
        input_nps.append(test_input)

    predictions = []

    t5_tokenizer = T5Tokenizer.from_pretrained(model)
    t5_config = T5Config.from_pretrained(model)
    t5_mlm = T5ForConditionalGeneration.from_pretrained(model, config=t5_config).to('cuda')

    _1_missing, _2_missing = 0, 0
    batch_size = 32
    num_batches = int(math.ceil(len(prompts)/batch_size))
    for bindx in tqdm(range(num_batches)):
        b_prompts = prompts[batch_size*bindx:batch_size*(bindx+1)]
        b_input_nps = input_nps[batch_size*bindx:batch_size*(bindx+1)]
        
        encoded = t5_tokenizer.batch_encode_plus(b_prompts, 
                                                add_special_tokens=True, 
                                                return_tensors='pt',
                                                padding=True)
        input_ids = encoded['input_ids'].to('cuda')
        outputs = t5_mlm.generate(input_ids=input_ids, 
                    num_beams=beam_size, 
                    num_return_sequences=num_return_sequences,
                    max_length=20)

        for output, input_np in zip(outputs, b_input_nps):
            _1_noun, _2_noun = input_np.split()
            prediction = t5_tokenizer.decode(output[2:], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            if '<extra_id_1>' in prediction:
                _1_index = prediction.index('<extra_id_1>')
                p1 = prediction[:_1_index]
                if '<extra_id_2>' in prediction:
                    _2_index = prediction.index('<extra_id_2>')
                    p2 = prediction[_1_index+12:_2_index].strip('.').strip()
                else:
                    _1_missing += 1
                    p2 = ''
            else:
                print('extra_id_1 is missing!', prediction, _2_missing)
                _2_missing += 1
                p1 = prediction
                p2 = ''
            if data == 'semeval':
                prediction = f'{_2_noun} {p1} {_1_noun}'
                predictions.append(f'{_1_noun} {_2_noun} is a '+prediction)
            else:
                if train_k_nn != 0:
                    prediction = f'{input_np} {p1} {_1_noun} {p2}'.strip()
                else:
                    prediction = f'{input_np} is a {p1} the {_1_noun} {p2}'.strip()
                predictions.append(prediction)

            # if data == 'semeval':
            #     of.write(_1_noun+'\t'+_2_noun+'\t'+prediction+'\t1\n')
        # inpf.write(prompt+'\n\n')

metrics = run_summarization.compute_metrics_helper(predictions, test_dataset['outputs'], None, test=True)
print('BLEU, BLEURT, NLI: ', metrics['bleu_scores'], metrics['bleurt_scores'], metrics['nli_scores'])
