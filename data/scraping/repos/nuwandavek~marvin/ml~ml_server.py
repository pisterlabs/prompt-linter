# import flask related modules
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

# basic imports
import json
import sys
import os 

# Pytorch imports
import torch
from torchtext.data.utils import get_tokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelWithLMHead, AutoConfig, HfArgumentParser

# Joint Model imports
from jointclassifier.joint_args import ModelArguments, DataTrainingArguments, TrainingArguments
from jointclassifier.joint_dataloader import load_dataset
from jointclassifier.joint_trainer import JointTrainer
from jointclassifier.single_trainer import SingleTrainer
from jointclassifier.joint_model_v1 import JointSeqClassifier

#Utils and constants
from constants import MODEL_PATHS
from utils import get_buckets, bucket_match, sort_results, filter_results

import openai
import argparse

app = Flask(__name__)
CORS(app)



# def load_models(mode):
#     global classifier_tokenizer, classifier_trainer, classifier_model, transfer_model, transfer_tokenizer, transfer_model_shake, transfer_model_abs, transfer_model_wiki
#     if mode in ['micro-formality','micro-joint','macro-shakespeare']:
#         transfer_model_shake = None
#         transfer_model_abs = None
#         transfer_model_wiki = None
        
#         mode_paths = MODEL_PATHS[mode]
#         model_args = ModelArguments(
#             model_name_or_path=mode_paths['classifier_name'],
#             model_nick=mode_paths['classifier_nick'],
#             cache_dir="./models/cache"
#         )

#         data_args = DataTrainingArguments(
#             max_seq_len=64,
#             task=mode_paths['classifier_task']
#         )

#         training_args = TrainingArguments(
#             output_dir = mode_paths['classifier'],
#             train_jointly= True
#         )
#         idx_to_classes = mode_paths['idx_to_classes']

#         label_dims = mode_paths['label_dims']

#         classifier_model = JointSeqClassifier.from_pretrained(
#             training_args.output_dir,
#             tasks=data_args.task.split('+'),
#             model_args=model_args,
#             task_if_single=None, 
#             joint = training_args.train_jointly,
#             label_dims=label_dims
#         )
#         classifier_trainer = JointTrainer(
#             [training_args,model_args, data_args], 
#             classifier_model, idx_to_classes = idx_to_classes
#         )
#         classifier_tokenizer = AutoTokenizer.from_pretrained(
#             model_args.model_name_or_path, 
#             cache_dir=model_args.cache_dir,
#             model_max_length = data_args.max_seq_len
#         )

#         transfer_tokenizer = AutoTokenizer.from_pretrained(mode_paths['transfer_name'])
#         transfer_model = AutoModelWithLMHead.from_pretrained(mode_paths['transfer'])   
#     elif mode in ['macro-binary']:
#         classifier_model = None
#         transfer_model = None
#         mode_paths = MODEL_PATHS[mode]

#         transfer_tokenizer = AutoTokenizer.from_pretrained(mode_paths['transfer_name'])
#         transfer_model_shake = AutoModelWithLMHead.from_pretrained(mode_paths['transfer_shake'])
#         transfer_model_abs = AutoModelWithLMHead.from_pretrained(mode_paths['transfer_abs'])
#         transfer_model_wiki = AutoModelWithLMHead.from_pretrained(mode_paths['transfer_wiki'])
        

def load_models(modes):
    global classifier_tokenizer, classifier_trainers, classifier_models, transfer_models, transfer_tokenizer
    classifier_models= {}
    classifier_trainers = {}
    transfer_models = {}
    transfer_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATHS['common']['transfer_name'], model_max_length=64, cache_dir="./models/cache")
    classifier_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATHS['common']['classifier_name'], model_max_length=64, cache_dir="./models/cache")
    for mode in modes:
        if mode in ['micro-formality','macro-shakespeare']:
            mode_paths = MODEL_PATHS[mode]
            model_args = ModelArguments(
                model_name_or_path=mode_paths['classifier_name'],
                model_nick=mode_paths['classifier_nick'],
                cache_dir="./models/cache"
            )

            data_args = DataTrainingArguments(
                max_seq_len=64,
                task=mode_paths['classifier_task']
            )

            training_args = TrainingArguments(
                output_dir = mode_paths['classifier'],
                train_jointly= True
            )
            idx_to_classes = mode_paths['idx_to_classes']

            label_dims = mode_paths['label_dims']

            classifier_models[mode] = JointSeqClassifier.from_pretrained(
                training_args.output_dir,
                tasks=data_args.task.split('+'),
                model_args=model_args,
                task_if_single=None, 
                joint = training_args.train_jointly,
                label_dims=label_dims
            )
            classifier_trainers[mode] = JointTrainer(
                [training_args,model_args, data_args], 
                classifier_models[mode], idx_to_classes = idx_to_classes
            )
            

            transfer_models[mode] = AutoModelWithLMHead.from_pretrained(mode_paths['transfer'])   
        elif mode in ['macro-binary']:
            mode_paths = MODEL_PATHS[mode]

            transfer_models[mode+"-shake"] = AutoModelWithLMHead.from_pretrained(mode_paths['transfer_shake'])
            transfer_models[mode+"-abs"] = AutoModelWithLMHead.from_pretrained(mode_paths['transfer_abs'])
            transfer_models[mode+"-wiki"] = AutoModelWithLMHead.from_pretrained(mode_paths['transfer_wiki'])
        
        elif mode in ['micro-joint']:
            mode_paths = MODEL_PATHS[mode]
            transfer_models[mode] = AutoModelWithLMHead.from_pretrained(mode_paths['transfer'])
                

@app.route("/hello")
def hello():
    res = {
        "world": 42,
        "app": "ml"
    }
    return res


@app.route("/swap_models", methods=['POST'])
def swap_models():
    mode = request.args.get('mode', type = str)
    print(mode)
    try:
        load_models(mode)
    except Exception as e:
       print(e)
       return {'message' : 'Models Swap Failure! :('}, 500
 
    return {'message' : 'Models Swap Success! :)'}, 200


@app.route('/classification', methods = ['GET'])
def get_joint_classify_and_salience():
    '''
    Inputs:
    Input is assumed to be json of the form 
      {text: "some text"}.
  
    Results:
      Run ML classification model on text. 
      
    Returns:
      res: a dict containing information on 
        classification and input salience weights.
        It has a key 'tokens' which is an array of the 
        tokenized input text. It also has a key for each 
        classification task. Each of these are themselves
        dicts containing keys for the predicted class, 
        the probability of this class, and also the salience score
        for each token from the tokenized input. 
    '''
    # Get text input from request
    text = request.args.get('text', type = str)
    text = text.strip()
    lower = text.lower()
    mode = request.args.get('mode', type = str)
    tokens = []
    sentence_seen = 0

    joint_tokens = classifier_tokenizer.convert_ids_to_tokens(classifier_tokenizer.encode(lower))[1:-1]
    for token in joint_tokens:
        # Handle case where the tokenizer splits some suffix as it's own token
        if len(token) > 2:
          if token[:2] == '##':
            token = token[2:]
        occ = lower[sentence_seen:].find(token)
        start = occ + sentence_seen
        end = start + len(token)
        adj_len = len(token)
        sentence_seen = sentence_seen + adj_len + occ
        tokens.append({'text' : text[start:end], 'start' : start, 'end' : end})
    
    
    if mode=='micro-joint':
        res = classifier_trainers['micro-formality'].predict_for_sentence(lower, classifier_tokenizer, salience=True)
    else:
        res = classifier_trainers[mode].predict_for_sentence(lower, classifier_tokenizer, salience=True)
    res['tokens'] = tokens
    return res, 200

@app.route('/transfer', methods = ['GET'])
def get_transfer():
    # Get text input from request
    text = request.args.get('text', type = str)
    mode = request.args.get('mode', type = str)
    controls = request.args.get('controls', type = str)
    text = text.strip()
    # lower = text.lower()
    lower = text
    controls = json.loads(controls)

    print(controls)
    controls['suggestions'] = int(min(5,max(1,float(controls['suggestions']))))
    if mode=="micro-formality":
        classifier_output = classifier_trainers[mode].predict_for_sentence(lower, classifier_tokenizer, salience=False)
        input_bucket = get_buckets(float(classifier_output['formality']['prob']), 'formality')
        output_bucket = ['low', 'mid', 'high'][int(controls['formality'])]
        transfer_input = "transfer: "+lower+' | input: '+input_bucket + ' | output: '+output_bucket

        t = transfer_tokenizer(transfer_input, return_tensors='pt')
        gen = transfer_models[mode].generate(input_ids= t.input_ids, attention_mask = t.attention_mask, max_length=70, 
                                            num_beams=15,
                                            #    early_stopping=True,
                                            encoder_no_repeat_ngram_size=5,
                                            no_repeat_ngram_size=3,
                                            num_beam_groups=5,
                                            diversity_penalty=0.5,
                                            # num_return_sequences=int(controls['suggestions'])
                                            num_return_sequences=10
                                            )
        transfers = transfer_tokenizer.batch_decode(gen, skip_special_tokens=True)

        res = {
            'input' : {
                'text' : text,
                'probs' : {
                    'formality' : classifier_output['formality']['prob']
                },
            },
            "goal" : f"Formality : {output_bucket}",
        }
        suggestions = []
        for transfer in transfers:
            cls_opt = classifier_trainers[mode].predict_for_sentence(transfer, classifier_tokenizer, salience=False)
            temp = {
                'text' : transfer,
                'probs' : {
                    'formality' : cls_opt['formality']['prob']
                }
            }
            suggestions.append(temp)

        suggestions = filter_results(suggestions, ['formality'], [output_bucket])
        suggestions = sort_results(suggestions, ['formality'], [output_bucket])
        res['suggestions'] = suggestions[:int(controls['suggestions'])]
        
        if output_bucket=='high' and server_args.openai:
            oai = get_openai_result(text)
            cls_opt = classifier_trainers[mode].predict_for_sentence(transfer, classifier_tokenizer, salience=False)
            temp = {
                'text' : oai,
                'probs' : {
                    'formality' : cls_opt['formality']['prob']
                }
            }
            res['openai'] = temp
        else:
            res['openai'] = {}
        
    elif mode=="macro-shakespeare":
        classifier_output = classifier_trainers[mode].predict_for_sentence(lower, classifier_tokenizer, salience=False)
        input_bucket = get_buckets(float(classifier_output['shakespeare']['prob']), 'shakespeare')
        output_bucket = ['low', 'mid', 'high'][int(controls['shakespeare'])]
        transfer_input = "transfer: "+lower+' | input: '+input_bucket + ' | output: '+output_bucket

        t = transfer_tokenizer(transfer_input, return_tensors='pt')
        gen = transfer_models[mode].generate(input_ids= t.input_ids, attention_mask = t.attention_mask, max_length=70, 
                                            num_beams=15,
                                            #    early_stopping=True,
                                            encoder_no_repeat_ngram_size=5,
                                            no_repeat_ngram_size=3,
                                            num_beam_groups=5,
                                            diversity_penalty=0.5,
                                            # num_return_sequences=int(controls['suggestions'])
                                            num_return_sequences=10
                                            )
        transfers = transfer_tokenizer.batch_decode(gen, skip_special_tokens=True)

        res = {
            'input' : {
                'text' : text,
                'probs' : {
                    'shakespeare' : classifier_output['shakespeare']['prob']
                },
            },
            "goal" : f"Shakespeare : {output_bucket}",
            "suggestions":[],
            "openai":{}
        }
        suggestions = []
        for transfer in transfers:
            cls_opt = classifier_trainers[mode].predict_for_sentence(transfer, classifier_tokenizer, salience=False)
            temp = {
                'text' : transfer,
                'probs' : {
                    'shakespeare' : cls_opt['shakespeare']['prob']
                }
            }
            suggestions.append(temp)

        suggestions = filter_results(suggestions, ['shakespeare'], [output_bucket])
        suggestions = sort_results(suggestions, ['shakespeare'], [output_bucket])
        res['suggestions'] = suggestions[:int(controls['suggestions'])]
        
    elif mode=="micro-joint":
        classifier_output = classifier_trainers['micro-formality'].predict_for_sentence(lower, classifier_tokenizer, salience=False)
        input_bucket_f = get_buckets(float(classifier_output['formality']['prob']), 'formality')
        input_bucket_e = get_buckets(float(classifier_output['emo']['prob']), 'emo')
        output_bucket_f = ['low', 'mid', 'high'][int(controls['formality'])]
        output_bucket_e = ['low', 'mid', 'high'][int(controls['emo'])]
        transfer_input = 'transfer: ' + lower + ' | input formality: '+input_bucket_f + ' | input emotion: '+input_bucket_e +' | output formality: '+output_bucket_f +' | output emotion: '+output_bucket_e

        print('\n\n',transfer_input,'\n\n')
        
        t = transfer_tokenizer(transfer_input, return_tensors='pt')
        gen = transfer_models[mode].generate(input_ids= t.input_ids, attention_mask = t.attention_mask, max_length=70, 
                                            num_beams=15,
                                            #    early_stopping=True,
                                            encoder_no_repeat_ngram_size=5,
                                            no_repeat_ngram_size=3,
                                            num_beam_groups=5,
                                            diversity_penalty=0.5,
                                            num_return_sequences=10
                                            # num_return_sequences=int(controls['suggestions'])
                                            )
        transfers = transfer_tokenizer.batch_decode(gen, skip_special_tokens=True)

        res = {
            'input' : {
                'text' : text,
                'probs' : {
                    'formality' : classifier_output['formality']['prob'],
                    'emo' : classifier_output['emo']['prob']
                },
            },
            "goal" : f"Formality : {output_bucket_f}; Emotion : {output_bucket_e}",
            "suggestions":[],
            "openai":{}
        }
        suggestions = []
        for transfer in transfers:
            cls_opt = classifier_trainers['micro-formality'].predict_for_sentence(transfer, classifier_tokenizer, salience=False)
            temp = {
                'text' : transfer,
                'probs' : {
                    'formality' : cls_opt['formality']['prob'],
                    'emo' : cls_opt['emo']['prob']
                }
            }
            suggestions.append(temp)
        suggestions = filter_results(suggestions, ['formality','emo'], [output_bucket_f, output_bucket_e])
        suggestions = sort_results(suggestions,  ['formality','emo'],  [output_bucket_f, output_bucket_e])
        res['suggestions'] = suggestions[:int(controls['suggestions'])]
        
        
    elif mode=="macro-binary":
        transfer_input = 'transfer: ' + lower
        print('\n\n',transfer_input,'\n\n')
        t = transfer_tokenizer(transfer_input, return_tensors='pt')
        
        if int(controls['macro']) == 0:
            gen = transfer_models[mode+'-wiki'].generate(input_ids= t.input_ids, attention_mask = t.attention_mask, max_length=70, 
                                            num_beams=12,
                                            #    early_stopping=True,
                                            encoder_no_repeat_ngram_size=5,
                                            no_repeat_ngram_size=3,
                                            num_beam_groups=3,
                                            diversity_penalty=0.5,
                                            num_return_sequences=int(controls['suggestions'])
                                            )
        elif int(controls['macro']) == 1:
            gen = transfer_models[mode+'-shake'].generate(input_ids= t.input_ids, attention_mask = t.attention_mask, max_length=70, 
                                            num_beams=12,
                                            #    early_stopping=True,
                                            encoder_no_repeat_ngram_size=5,
                                            no_repeat_ngram_size=3,
                                            num_beam_groups=3,
                                            diversity_penalty=0.5,
                                            num_return_sequences=int(controls['suggestions'])
                                            )
        elif int(controls['macro']) == 2:
            gen = transfer_models[mode+'-abs'].generate(input_ids= t.input_ids, attention_mask = t.attention_mask, max_length=70, 
                                            num_beams=12,
                                            #    early_stopping=True,
                                            encoder_no_repeat_ngram_size=5,
                                            no_repeat_ngram_size=3,
                                            num_beam_groups=3,
                                            diversity_penalty=0.5,
                                            num_return_sequences=int(controls['suggestions'])
                                            )
         
        transfers = transfer_tokenizer.batch_decode(gen, skip_special_tokens=True)

        res = {
            'input' : {
                'text' : text,
            },
            "goal" : ["Wikipedia", "Shakespeare", "Scientific Abstract"][int(controls['macro'])],
            "suggestions":[],
            "openai":{}
        }
        for transfer in transfers:
            temp = {
                'text' : transfer,
            }
            res['suggestions'].append(temp)
    return res, 200

def load_openai_key():
    with open("./key.txt") as fob:
        openai.api_key = fob.read().strip()

def get_openai_result(text):
   prompt = "Plain Language: what're u doin?\nFormal Language: What are you doing?\nPlain Language: what's up?\nFormal Language: What is up?\nPlain Language: i wanna eat ice cream today!\nFormal Language: I want to eat ice cream today.\nPlain Language: wtf is his problem?\nFormal Language: What is his issue?\nPlain Language: i feel bummed about the store shutting down.\nFormal Language: I feel unhappy about the store closing.\nPlain Language: "

   prompt = prompt + text + "\nFormal Language:"
   res = openai.Completion.create(
       engine="davinci",
      prompt= prompt,
       max_tokens=64,
       temperature=0.15,
       stop="\n"
   )

   return res.choices[0].text.strip()

if __name__ == '__main__':
    load_models(['micro-formality','macro-shakespeare','micro-joint','macro-binary'])
    # print(transfer_models.keys())
    parser = argparse.ArgumentParser()
    parser.add_argument('--openai', help='Use openai API or not', default=False)
    global server_args
    server_args = parser.parse_args()

    if server_args.openai==True:
        load_openai_key()
    
    app.run(host="0.0.0.0", port=5001)
