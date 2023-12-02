# fine-tuning model on citation classification
import os
import argparse
import torch
import logging
import csv
csv.field_size_limit(500 * 1024 * 1024)
import datetime
import random
from tqdm import trange, tqdm
from transformers import OpenAIGPTTokenizer, OpenAIGPTModel

import sys
sys.path.append('/home/chenqun/projects/fy/dynamic_citation_context/')

import numpy as np
from sklearn.metrics import f1_score
from classification_utils import processors_fixed_context, output_modes, CITATION_CLASSIFICATION_NUM_LABELS, \
    convert_examples_to_features, processors_dynamic_context
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, DistributedSampler
from data import DATA_PROCESSED_DIR
from transformers import AdamW
from transformers import (AutoTokenizer, AutoModel)

logger = logging.getLogger(__name__)
# tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')
#LMModel = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")


# tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
# LMModel = AutoModel.from_pretrained(args.model_name_or_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LMClass(torch.nn.Module):
    def __init__(self, model_path):
        super(LMClass, self).__init__()
        # self.model = AutoModel.from_pretrained(model_path)
        self.model = OpenAIGPTModel.from_pretrained(model_path)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, 6)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        outputs = self.model(input_ids, token_type_ids, attention_mask)
        
        hidden_state1 = outputs[0]
        # print (hidden_state1.shape)
        last_hidden_state_cls = torch.mean(hidden_state1, dim=1)
        pooler = self.dropout(last_hidden_state_cls)
        output = self.classifier(pooler)

        return output


def set_seed(seed_value=42):
    # Set seed for reproducibility.

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def calcuate_accu(big_idx, targets):
    n_correct = (big_idx == targets).sum().item()
    return n_correct


def train(args, model, loss_function, optimizer, train_dataloader, eval_dataloader=None):
    if not eval_dataloader:
        eval_results = open(os.path.join(args.output_dir, args.context_window, 'results_dev.txt'), 'w+')
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = args.model_name_or_path.split('/')[0]
    # check if output directory exists
    if not os.path.exists(os.path.join(args.output_dir, args.context_window)):
        os.makedirs(os.path.join(args.output_dir, args.context_window))

    args.snapshot_path = os.path.join(args.output_dir, args.context_window,
                                      '{}_{}_{}.pt'.format(model_name, args.max_seq_length, timestamp))

    epoch = 0
    best_dev_f1, unimproved_iters = 0, 0
    # eval_results = open(os.path.join(args.output_dir, args.context_window, 'results_dev.txt'),'w+')
    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        epoch += 1
        model.train()
        tr_loss, global_step = 0, 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            # loss, _ = model(input_ids, segment_ids, input_mask, label_ids)
            outputs = model(input_ids, segment_ids, input_mask, label_ids)
            # loss = loss_function(outputs, torch.argmax(label_ids, dim=1))
            loss = loss_function(outputs, label_ids)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
           
            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()
                global_step += 1

        # validate
        if args.do_eval:

            eval_loss, eval_accuracy, macro_f1, micro_f1 = evaluate(args, model, eval_dataloader, loss_function)
            result = {'eval_loss': eval_loss,
                      'eval_accuracy': eval_accuracy,
                      'global_step': global_step,
                      'loss': tr_loss / nb_tr_steps,
                      'macro_f1': macro_f1,
                      'micro_f1': micro_f1
                      }

            eval_results.write("Epoch: {}".format(epoch))
            eval_results.write("\n")
            eval_results.write("Epoch results: {}".format(result))
            eval_results.write("\n")

            # Update validation results
            if macro_f1 > best_dev_f1:
                unimproved_iters = 0
                best_dev_f1 = macro_f1
                torch.save(model.state_dict(), args.snapshot_path)

            else:
                unimproved_iters += 1
                if unimproved_iters >= args.patience:
                    early_stop = True
                    eval_results.write("Early Stopping. Epoch: {}, Best Dev F1: {}".format(epoch, best_dev_f1))
                    break

        eval_results.close()

    return


def evaluate(args, model, eval_dataloader, loss_function, split=None):
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    predicted_labels, target_labels = list(), list()
    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluate"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, label_ids)
            tmp_eval_loss = loss_function(logits, label_ids)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()

        outputs = np.argmax(logits, axis=1)
        predicted_labels.extend(outputs)
        target_labels.extend(label_ids)

        tmp_eval_accuracy = np.sum(outputs == label_ids)
        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples
    macro_f1 = f1_score(target_labels, predicted_labels, average='macro')
    micro_f1 = f1_score(target_labels, predicted_labels, average='micro')

    if split == 'test':
        return eval_loss, eval_accuracy, macro_f1, micro_f1, predicted_labels, target_labels

    else:
        return eval_loss, eval_accuracy, macro_f1, micro_f1


def load_and_cache_examples(args, tokenizer, processor, label_list, evaluate, test):

    # Load data features from cache or dataset file
    if evaluate and not test:
        split = 'dev'
    elif test:
        split = 'test'
    else:
        split = 'train'

    logger.info("Creating features from dataset file at %s", args.data_dir)

    if split == 'dev':
        print('Getting validation examples')
        examples = processor.get_dev_examples(args.data_dir)
        print('collected valid')
    elif split == 'test':
        print('Getting testing examples')
        examples = processor.get_test_examples(args.data_dir)
    else:
        print('Getting training examples')
        examples = processor.get_train_examples(args.data_dir)

    features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, print_examples=True)

    unpadded_input_ids = [f.input_ids for f in features]
    unpadded_input_mask = [f.input_mask for f in features]
    unpadded_segment_ids = [f.segment_ids for f in features]

    padded_input_ids = torch.tensor(unpadded_input_ids, dtype=torch.long)
    padded_input_mask = torch.tensor(unpadded_input_mask, dtype=torch.long)
    padded_segment_ids = torch.tensor(unpadded_segment_ids, dtype=torch.long)
    label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    # tensors = [padded_input_ids, padded_input_mask, padded_segment_ids, label_ids]

    tensor_dataset = TensorDataset(padded_input_ids, padded_input_mask, padded_segment_ids, label_ids)

    return tensor_dataset


def main():
    parser = argparse.ArgumentParser(description="PyTorch code for citation classification")
    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir for the task.")
    parser.add_argument("--data_set", default='sdp_act', type=str, required=False,
                        help="The data set used for classification - sdp_act or acl_arc")
    parser.add_argument("--context_type", default='non_contiguous', type=str, required=False,
                        help="The citation context used - 1. fixed_context, 2. contiguous or 3. non_contiguous.")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=False,
                        help="Path to pre-trained model")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="Name of the task - citation_function/citation_influence")
    parser.add_argument("--context_window", default=None, type=str, required=True,
                        help="Fixed/dynamic context window size on which training/testing has to be done.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--snapshot_path", default="", type=str,
                        help="Path to saved model")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--per_gpu_test_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for testing.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )

    parser.add_argument('--warmup-proportion',
                        default=0.1,
                        type=float,
                        help='Proportion of training to perform linear learning rate warmup for')

    parser.add_argument('--gradient-accumulation-steps',
                        type=int,
                        default=1,
                        help='Number of updates steps to accumulate before performing a backward/update pass')

    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    args = parser.parse_args()

    # device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    print('Device:', str(device).upper())
    print('Number of GPUs:', args.n_gpu)

    args.device = device
    set_seed(args.seed)

    args.context_window = args.context_window.lower()


    print (DATA_PROCESSED_DIR)


    # input_dir = DATA_PROCESSED_DIR / f"{args.citation_context}_{args.data_set}/"

    # print (input_dir)

    # if not args.data_dir == input_dir:
    #     raise ValueError('Please generate the citation context before proceeding to classification')

    print (args.context_window)
    if args.context_window not in processors_dynamic_context:
        #raise ValueError('Fixed context window size: %s not found', args.context_window)
        processors = processors_fixed_context[args.context_window]()
    else:
        print ('debug1')
        processors = processors_dynamic_context[args.context_window]()

    args.output_mode = output_modes[args.task_name]
    label_list = processors.get_labels()
    args.num_labels = CITATION_CLASSIFICATION_NUM_LABELS[args.task_name]

    # check if output directory exists
    if not os.path.exists(os.path.join(args.output_dir, args.context_window)):
        os.makedirs(os.path.join(args.output_dir, args.context_window))

    # args.model_type = args.model_type.lower()
    model = LMClass(args.model_name_or_path)

    model.to(args.device)

    to_write_predicted = list()
    to_write_actual = list()

    logger.info("Training/evaluation parameters %s", args)
    train_tensor = load_and_cache_examples(args, tokenizer, processors, label_list,
                                           evaluate=False, test=False)
    dev_tensor = load_and_cache_examples(args, tokenizer, processors, label_list,
                                         evaluate=True, test=False)
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    args.test_batch_size = args.per_gpu_test_batch_size * max(1, args.n_gpu)

    
    print (train_tensor)

    full_train_data = torch.utils.data.ConcatDataset([train_tensor, dev_tensor])
    train_sampler = RandomSampler(full_train_data)
    train_dataloader = DataLoader(full_train_data, sampler=train_sampler, batch_size=args.test_batch_size)

    # if 'sdp_act' in args.data_dir:
    weights = [0.30435841, 1.34843581, 2.91375291, 7.57575758, 1.78062678, 1.06837607] #sdp_act
    # else:
    #     weights = [0.32256169, 0.92424242, 4.65254237, 4.81578947, 3.8125, 0.88263666]  # acl_arc
    
    print (weights)
    
    # weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    class_weights = torch.FloatTensor(weights).to(device)
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)
    no_decay = ['bias', 'gamma', 'beta']

    optimizer_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    optimizer = AdamW(optimizer_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    # Training
    train(args, model, loss_function, optimizer, train_dataloader)

    # Evaluation
    test_tensor = load_and_cache_examples(args, tokenizer, processors, label_list,
                                          evaluate=True, test=True)

    if args.local_rank == -1:
        testdata_sampler = RandomSampler(test_tensor)
    else:
        testdata_sampler = DistributedSampler(test_tensor)

    test_dataloader = DataLoader(test_tensor, sampler=testdata_sampler, batch_size=args.test_batch_size)
    output_model_file = os.path.join(args.output_dir, args.context_window, 'pytorch_model.bin')
    torch.save(model.state_dict(), output_model_file)
    eval_loss, eval_accuracy, macro_f1, micro_f1, predicted_labels, target_labels = evaluate(args, model,
                                                                                             test_dataloader,
                                                                                             loss_function,
                                                                                             split='test')
    ground_truth_file = open(os.path.join(args.output_dir, args.context_window, 'ground_truth.csv'), 'w+')
    prediction_file = open(os.path.join(args.output_dir, args.context_window, 'prediction.csv'), 'w+')

    for idx1, pr in enumerate(predicted_labels):
        to_write_predicted.append(['CCT' + str(idx1 + 1), pr])

    prediction_file.write('unique_id,citation_class_label' + '\n')
    predictions = csv.writer(prediction_file)
    predictions.writerows(to_write_predicted)
    prediction_file.close()

    for idx2, tr in enumerate(target_labels):
        to_write_actual.append(['CCT' + str(idx2 + 1), tr])

    ground_truth_file.write('unique_id,citation_class_label' + '\n')
    actual = csv.writer(ground_truth_file)
    actual.writerows(to_write_actual)
    ground_truth_file.close()

    test_results = open(os.path.join(args.output_dir, args.context_window, 'results_test.txt'), 'w+')
    test_results.write("Testdata Results:" + '\n')
    test_results.write("Macro F-Score: {}".format(macro_f1) + '\n')
    test_results.write("Micro F-Score: {}".format(micro_f1) + '\n')
    test_results.close()


if __name__ == "__main__":
    main()