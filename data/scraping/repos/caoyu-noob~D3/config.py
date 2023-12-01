from attrdict import AttrDict
from copy import deepcopy
import torch
from model.utils import openai_transformer_config
import git
import argparse


repo = git.Repo(search_parent_directories=True)


def cast2(type_):
    return lambda val: val if val is None else type_(val)


def get_model_config(args):
    default_config = openai_transformer_config()
    config = AttrDict({'bpe_vocab_path': './parameters/bpe.vocab',
                       'bpe_codes_path': './parameters/bpe.code',
                       'checkpoint_path': './checkpoints/last_checkpoint',  # Keep the checpoint folder for the checkpoints of the agents
                       'n_layers': default_config.n_layers,
                       'n_pos_embeddings': 512,
                       'embeddings_size': default_config.embeddings_size,
                       'n_heads': default_config.n_heads,
                       'dropout': default_config.dropout,
                       'embed_dropout': default_config.embed_dropout,
                       'attn_dropout': default_config.attn_dropout,
                       'ff_dropout': default_config.ff_dropout,
                       'normalize_embeddings': args.normalize_embeddings,
                       'max_seq_len': 128,
                       'beam_size': args.beam_size,
                       'diversity_coef': args.diversity_coef,
                       'diversity_groups': args.diversity_groups,
                       'annealing_topk': args.annealing_topk,
                       'annealing': args.annealing,
                       'length_penalty': args.length_penalty,
                       'n_segments': None,
                       'constant_embedding': args.constant_embedding,
                       'multiple_choice_head': args.multiple_choice_head,
                       'share_models': True,
                       'successive_attention': args.successive_attention,
                       'sparse_embeddings': args.sparse_embeddings,
                       'shared_attention': args.shared_attention,
                       'dialog_embeddings': args.dialog_embeddings,
                       'single_input': args.single_input,
                       'use_start_end': args.use_start_end,
                       'apex_level': args.apex_level,  # 'O0', 'O1', 'O2', 'O3',
                       'bs_temperature': args.bs_temperature,
                       'bs_nucleus_p': args.bs_nucleus_p,
                       'same_embedding_lm': args.same_embedding_lm,
                       })

    return config


def get_trainer_config(args, curriculum_config=False):
    config = AttrDict({'n_epochs': args.n_epochs,
                       'writer_comment': args.writer_comment,
                       'train_batch_size': args.train_batch_size,
                       'meta_batch_size': args.meta_batch_size,
                       'batch_split': args.batch_split,
                       'test_batch_size': args.test_batch_size,
                       'lr': args.lr,
                       'lr_warmup': args.lr_warmup,  # a fraction of total training (epoch * train_set_length) if linear_schedule == True
                       'weight_decay': 0.01,
                       's2s_weight': args.s2s_weight,
                       'lm_weight': args.lm_weight,
                       'risk_weight': args.risk_weight,
                       'hits_weight': args.hits_weight,
                       'negative_samples': args.negative_samples,
                       'n_jobs': 4,
                       'label_smoothing': args.label_smoothing,
                       'clip_grad': args.clip_grad,
                       'alpha_clip_grad': args.alpha_clip_grad,
                       'test_period': 1,
                       'seed': args.seed,
                       'device': 'cuda',
                       'zero_shot': args.zero_shot,
                       'persona_augment': args.persona_augment,
                       'persona_aug_syn_proba': args.persona_aug_syn_proba,
                       'apex_loss_scale': args.apex_loss_scale, # e.g. '128', 'dynamic'
                       'linear_schedule': args.linear_schedule,
                       'evaluate_full_sequences': args.evaluate_full_sequences,
                       'limit_eval_size': args.limit_eval_size,
                       'limit_train_size': args.limit_train_size,
                       'risk_metric': args.risk_metric,
                       'load_last': args.load_last, #./checkpoints/last_checkpoint',  # Now that we save several experiments you can put the path of the checpoint file you want to load here
                       'load_alpha_last': args.load_alpha_last,
                       'repo_id': str(repo),
                       'repo_sha': str(repo.head.object.hexsha),
                       'repo_branch': str(repo.active_branch),
                       'openai_parameters_dir': './parameters',
                       'last_checkpoint_path': 'last_checkpoint',  # there are now in the ./runs/XXX/ experiments folders
                       'eval_references_file': 'eval_references_file',
                       'eval_predictions_file': 'eval_predictions_file',
                       'test_references_file': 'test_references_file',
                       'test_predictions_file_best': 'test_predictions_file_best',
                       'test_predictions_file_last': 'test_predictions_file_last',
                       'interrupt_checkpoint_path': 'interrupt_checkpoint',  # there are now in the ./runs/XXX/ experiments folders
                       'train_datasets': args.train_datasets,
                       'train_datasets_cache': args.train_datasets_cache,
                       'test_datasets': args.test_datasets,
                       'test_datasets_cache': args.test_datasets_cache,
                       'valid_datasets': args.valid_datasets,
                       'valid_datasets_cache': args.valid_datasets_cache,
                       'full_input': args.full_input,
                       'single_input': args.single_input,
                       'max_history_size': args.max_history_size,
                       'model_saving_interval': args.model_saving_interval,
                       'patience': args.patience,
                       'mixup_cache': args.mixup_cache,
                       'data_type': args.data_type,
                       })
    if curriculum_config:
        config.train_datasets = args.curriculum_train_datasets
        config.train_datasets_cache = args.curriculum_train_datasets_cache
        config.valid_datasets = args.curriculum_valid_datasets
        config.valid_datasets_cache = args.curriculum_valid_datasets_cache
        config.test_datasets = args.curriculum_valid_datasets
        config.test_datasets_cache = args.curriculum_valid_datasets_cache
        config.lr = args.curriculum_lr
        config.n_epochs = args.curriculum_n_epochs
        config.patience = args.patience
        config.max_history_size = args.curriculum_max_history_size
        config.eval_references_file = 'eval_references_file_stage1'
        config.eval_predictions_file = 'eval_predictions_file_stage1'
        config.test_references_file = 'test_reference_file_stage1'
        config.test_predictions_file_best = 'test_predictions_file_stage1_best'
        config.test_predictions_file_last = 'test_predictions_file_stage1_last'
        config.data_type = args.curriculum_data_type

    local_config = deepcopy(config)
    local_config.train_batch_size = 16
    local_config.batch_split = 2
    local_config.test_batch_size = 4
    local_config.n_jobs = 0
    local_config.device = 'cpu'
    local_config.risk_weight = 0
    local_config.zero_shot = False
    local_config.fp16 = False
    # local_config.train_datasets_cache = './datasets/train_datasets_cache.bin'
    # local_config.test_datasets_cache = './datasets/test_datasets_cache.bin'

    return config if torch.cuda.is_available() else local_config

class InputConfig():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', type=int, default=0)
        parser.add_argument('--normalize_embeddings', action='store_true')
        parser.add_argument('--beam_size', default=3, type=int)
        parser.add_argument('--inference_mode', default='beam', type=str)
        parser.add_argument('--response_k', default=1, type=int)
        parser.add_argument('--diversity_coef', default=0, type=int)
        parser.add_argument('--lr', default=6.25e-5, type=float)
        parser.add_argument('--meta_lr', default=1e-4, type=float)
        parser.add_argument('--finetune_lr', default=2e-4, type=float)
        parser.add_argument('--finetune_epochs', default=3, type=int)
        parser.add_argument('--lr_warmup', default=0.002, type=float)
        parser.add_argument('--clip_grad', default=None, type=float)
        parser.add_argument('--diversity_groups', default=1, type=int)
        parser.add_argument('--annealing_topk', default=None, type=int)
        parser.add_argument('--annealing', default=0, type=float)
        parser.add_argument('--length_penalty', default=0.6, type=float)
        parser.add_argument('--bs_temperature', default=1, type=float)
        parser.add_argument('--bs_nucleus_p', default=0, type=float)
        parser.add_argument('--apex_level', default=None, type=str)
        parser.add_argument('--constant_embedding', action='store_true')
        parser.add_argument('--multiple_choice_head', action='store_true')
        parser.add_argument('--successive_attention', action='store_true')
        parser.add_argument('--sparse_embeddings', default=False, type=bool)
        parser.add_argument('--dialog_embeddings', default=True, type=bool)
        parser.add_argument('--single_input', action='store_true')
        parser.add_argument('--no_persona', action='store_true')
        parser.add_argument('--mixup', action='store_true')
        parser.add_argument('--use_start_end', action='store_true')
        parser.add_argument('--zero_shot', action='store_true')
        parser.add_argument('--persona_augment', action='store_true')
        parser.add_argument('--linear_schedule', default=True, type=bool)
        parser.add_argument('--evaluate_full_sequences', default=True, type=bool)
        parser.add_argument('--n_epochs', default=3, type=int)
        parser.add_argument('--patience', default=-1, type=int, help="the training patience if the dev result "
                                                                     "does not promote then training ends")
        parser.add_argument('--train_batch_size', default=128, type=int)
        parser.add_argument('--batch_split', default=32, type=int)
        parser.add_argument('--test_batch_size', default=8, type=int)
        parser.add_argument('--writer_comment', default='', type=str)
        parser.add_argument('--s2s_weight', default=2, type=float)
        parser.add_argument('--lm_weight', default=1, type=float)
        parser.add_argument('--risk_weight', default=0, type=float)
        parser.add_argument('--hits_weight', default=0, type=float)
        parser.add_argument('--label_smoothing', default=-1, type=float,
                            help='Config for Seq2Seq model, whether use label smoothing loss, -1 means no smoothing')
        parser.add_argument('--negative_samples', default=0, type=int)
        parser.add_argument('--persona_aug_syn_proba', default=0, type=float)
        parser.add_argument('--apex_loss_scale', default=None, type=str)
        parser.add_argument('--limit_eval_size', default=-1, type=int)
        parser.add_argument('--limit_train_size', default=-1, type=int)
        parser.add_argument('--risk_metric', default='f1', type=str)
        parser.add_argument('--load_last', default='', type=str)
        parser.add_argument('--load_alpha_last', default='', type=str)
        parser.add_argument('--data_type', default='persona', type=str, help='data set types, persona/emoji/daily')
        parser.add_argument('--test_data_type', default=None, type=str, help='data set types, persona/emoji/daily')
        parser.add_argument('--emb_dim', default=300, type=int, help='Config for Seq2Seq model')
        parser.add_argument('--hidden_dim', default=300, type=int, help='Config for Seq2Seq model')
        parser.add_argument('--num_layers', default=6, type=int, help='Config for Seq2Seq model')
        parser.add_argument('--heads', default=4, type=int, help='Config for Seq2Seq model')
        parser.add_argument('--depth_size', default=40, type=int, help='Config for Seq2Seq model')
        parser.add_argument('--filter_size', default=50, type=int, help='Config for Seq2Seq model')
        parser.add_argument('--pointer_gen', action='store_true', help='Config for Seq2Seq model')
        parser.add_argument('--pretrained_emb_file', default='./glove/glove.6B.300d.txt', type=str)
        parser.add_argument('--vocab_path', default='./datasets/persona_vocab.bin', type=str)
        parser.add_argument('--extend_exist_vocab', default=None, type=str)
        parser.add_argument('--train_datasets', default='datasets/ConvAI2/train_self_original.txt', type=str)
        parser.add_argument('--valid_datasets', default='datasets/ConvAI2/valid_self_original.txt', type=str)
        parser.add_argument('--test_datasets', default='datasets/ConvAI2/test_self_original.txt', type=str)
        parser.add_argument('--cache_vocab_path', default='datasets/ConvAI2/cached_vocab.pickle', type=str)
        parser.add_argument('--train_datasets_cache', default='datasets/train_cache.bin', type=str)
        parser.add_argument('--valid_datasets_cache', default='datasets/valid_cache.bin', type=str)
        parser.add_argument('--test_datasets_cache', default='datasets/test_cache.bin', type=str)
        parser.add_argument('--extra_train_path', default=None, type=str)
        parser.add_argument('--extra_data_type', default='persona', type=str)
        parser.add_argument('--extra_cvae_utterances_path', default=None, type=str, help='The path indicates the CVAE augmented utterances')
        parser.add_argument('--generate_file_name', default='generation.json', type=str, help='The saved json file name '
                          'when inference on entailment data')
        parser.add_argument('--mixup_cache', default='datasets/mixup_cache.bin', type=str)
        parser.add_argument('--mixup_mode', default='alternate', type=str, help='The mode to execute the mixup operation'
                            'alternate=no mixup for one batch and mix up for next batch iteratively'
                            'all=mixup for all training samples, random=randomly mixup several samples within a batch'
                            'while rest samples remains unmixed')
        parser.add_argument('--mixup_model_path', default='./fasttext/persona_50_cbow.bin', type=str)
        parser.add_argument('--mixup_candidate_th', default=0.4, type=float)
        parser.add_argument('--mixup_ratio', default=0.15, type=float)
        parser.add_argument('--mixup_soft_loss_weight', default=0, type=float)
        parser.add_argument('--bert_mixup', action='store_true', help='whether use bert model for mixup')
        parser.add_argument('--replace', action='store_true', help='whether use directly replace token in samples')
        parser.add_argument('--few_shot', action='store_true', help='whether do few-shot learning')
        parser.add_argument('--shot_num', type=int, default=5, help='The shot number for training')
        parser.add_argument('--train_task_map', type=str, default=None, help='The path of task map json file between '
                                                                            'persona id and sample ids')
        parser.add_argument('--valid_task_map', type=str, default=None, help='The path of task map json file between '
                                                                             'persona id and sample ids')
        parser.add_argument('--test_task_map', type=str, default=None, help='The path of task map json file between '
                                                                             'persona id and sample ids')
        parser.add_argument('--meta_batch_size', type=int, default=16, help='The meta batch size')
        parser.add_argument('--meta_batch_ratio', type=float, default=1, help='The training and evaluation sample number'
                                                                              ' ratio in each batch')
        parser.add_argument('--full_input', action='store_true', help='whether use the concatenated persona, history'
                                                                      ' and reply as the input ids')
        parser.add_argument('--max_history_size', type=int, default=-1, help='max history size in input ids')
        parser.add_argument('--same_embedding_lm', type=int, default=1, help='the embedding in transformer and the '
                                                                             'weight in the lm are the same')
        parser.add_argument('--uncertainty_loss', action='store_true', help='whether use uncertainty loss')
        # parser.add_argument('--attention_weight', action='store_true', help='Whether use the learnable weight for '
        #                                            'attention obtained from different source in decoder')
        parser.add_argument('--model_type', type=str, default='gpt', help='gpt/gpt2/se2seq/rnn-seq2seq')
        parser.add_argument('--model_saving_interval', type=int, default=10, help='model saving interval for seq2seq')
        parser.add_argument('--entail_score_refs_file', type=str, default=None, help='The persona and idx json file for each '
                       'utterance, if None no entailment score will be calculated')
        parser.add_argument('--entail_model_path', type=str, default='./analysis/roberta_nli',
                help='The persona and idx json file for each utterance, if None no entailment score will be calculated')
        parser.add_argument('--bert_score_model_path', type=str, default='../roberta_large', help='The model path '
                'indicate which model will be used for bertscore, if None then no bertscore is calculated')
        parser.add_argument('--baseline_path', type=str, default='./bert_score/rescale_baseline/en/roberta-large.tsv',
                            help='The file path for the bert score rescale baseline')
        parser.add_argument('--rescale_with_baseline', action='store_true',
                            help='Whether rescale the bert score')
        parser.add_argument('--ignore_sample_indices', type=str, default=None,
                            help='The json file indicating which samples will be ignored')
        parser.add_argument('--shared_module', type=int, default=1)
        parser.add_argument('--shared_attention', type=int, default=1)
        parser.add_argument('--attention_pooling_type', type=str, default='mean', help='the method to pool attention '
                'output from different source(mean/min/max/sw/dw/linear/att) '
                'sw=source level weight, dw=dimension level weight, linear=linear transform for concatenating,'
                'att=extra transformer attention layer to fuse attention output,'
                'dys=dynamic determine the scalar weight for each source by a linear layer'
                'dyd=dynamic determine the vector weight for each dimension for each source by a linear layer'
                'mdys=mutual determine the scalar weight for each source by a linear layer'
                'mdyd=mutual dynamic determine the vector weight for each dimension for each source by a linear layer')

        '''Configurations related to curriculum learning'''
        parser.add_argument('--curriculum_learning', action='store_true', help='Whether to do curriculum learning')
        parser.add_argument('--curriculum_train_datasets', type=str, default=None)
        parser.add_argument('--curriculum_valid_datasets', type=str, default=None)
        parser.add_argument('--curriculum_train_datasets_cache', type=str, default=None)
        parser.add_argument('--curriculum_valid_datasets_cache', type=str, default=None)
        parser.add_argument('--curriculum_lr', type=float, default=2e-4)
        parser.add_argument('--curriculum_n_epochs', type=int, default=50)
        parser.add_argument('--curriculum_patience', type=int, default=-1)
        parser.add_argument('--curriculum_max_history_size', type=int, default=3)
        parser.add_argument('--curriculum_data_type', default='entailment', type=str)
        parser.add_argument('--curriculum_reverse', action='store_true', help='Whether using the reverse order of training')

        parser.add_argument('--local_rank', type=int, default=-1, help="Distributed training.")
        parser.add_argument('--server_ip', type=str, default='', help="Used for debugging on GPU machine.")
        parser.add_argument('--server_port', type=str, default='', help="Used for debugging on GPU machine.")

        self.args = parser.parse_args()
