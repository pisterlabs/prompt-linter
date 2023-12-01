import os
import random

import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import pandas as pd
from toolz.sandbox import unzip
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import more_itertools

import config
from c_parser.ast_parser import parse_ast_code_graph
from common import torch_util, util
from common.problem_util import to_cuda
from common.logger import init_a_file_logger, info
from common.opt import OpenAIAdam
from common.pycparser_util import tokenize_by_clex_fn
from common.torch_util import create_sequence_length_mask, permute_last_dim_to_second, expand_tensor_sequence_len, \
    expand_tensor_sequence_to_same, DynamicDecoder, pad_last_dim_of_tensor_list
from common.util import data_loader, show_process_map, PaddedList, convert_one_token_ids_to_code, filter_token_ids, \
    compile_c_code_by_gcc, create_token_set, create_token_mask_by_token_set, generate_mask, queued_data_loader, \
    CustomerDataSet, OrderedList, create_effect_keyword_ids_set
from experiment.experiment_util import load_common_error_data, load_common_error_data_sample_100, \
    create_addition_error_data, create_copy_addition_data, load_deepfix_error_data, \
    load_fake_deepfix_dataset_iterate_error_data_sample_100, load_fake_deepfix_dataset_iterate_error_data
from experiment.parse_xy_util import parse_output_and_position_map
from model.base_attention import TransformEncoderModel, TransformDecoderModel, TrueMaskedMultiHeaderAttention, \
    register_hook, PositionWiseFeedForwardNet, is_nan, record_is_nan
from model.encoder_sample_model import GraphEncoder
from read_data.load_data_vocabulary import create_common_error_vocabulary
from seq2seq.models import EncoderRNN, DecoderRNN
from vocabulary.transform_vocabulary_and_parser import TransformVocabularyAndSLK
from vocabulary.word_vocabulary import Vocabulary
from common.constants import pre_defined_c_tokens, pre_defined_c_library_tokens


MAX_LENGTH = 500
IGNORE_TOKEN = -1
is_debug = False


class CCodeErrorDataSet(CustomerDataSet):
    def __init__(self,
                 data_df: pd.DataFrame,
                 vocabulary: Vocabulary,
                 set_type: str,
                 transform=None,
                 transformer_vocab_slk=None,
                 no_filter=False,
                 use_ast=False, ):
        # super().__init__(data_df, vocabulary, set_type, transform, no_filter)
        self.set_type = set_type
        self.transform = transform
        self.vocabulary = vocabulary
        self.transformer = transformer_vocab_slk
        self.use_ast = use_ast
        self.keyword_ids = create_effect_keyword_ids_set(vocabulary)
        if data_df is not None:
            if not no_filter:
                self.data_df = self.filter_df(data_df)
            else:
                self.data_df = data_df
            self._samples = [row for i, row in self.data_df.iterrows()]
            if self.transform:
                self._samples = show_process_map(self.transform, self._samples)
            # for s in self._samples:
            #     for k, v in s.items():
            #         print("{}:shape {}".format(k, np.array(v).shape))

    def filter_df(self, df):
        df = df[df['error_code_word_id'].map(lambda x: x is not None)]
        # print('CCodeErrorDataSet df before: {}'.format(len(df)))
        df = df[df['distance'].map(lambda x: x >= 0)]
        # print('CCodeErrorDataSet df after: {}'.format(len(df)))
        # print(self.data_df['error_code_word_id'])
        # df['error_code_word_id'].map(lambda x: print(type(x)))
        # print(df['error_code_word_id'].map(lambda x: print(x))))
        df = df[df['error_code_word_id'].map(lambda x: len(x) < MAX_LENGTH)]
        end_id = self.vocabulary.word_to_id(self.vocabulary.end_tokens[0])

        slk_count = 0
        error_count = 0
        not_in_mask_error = 0
        def filter_slk(ac_code_ids):
            res = None
            nonlocal slk_count, error_count, not_in_mask_error
            slk_count += 1
            if slk_count%100 == 0:
                print('slk count: {}, error count: {}, not in mask error: {}'.format(slk_count, error_count, not_in_mask_error))
            try:
                mask_list = self.transformer.get_all_token_mask_train([ac_code_ids])[0]
                res = True
                for one_id, one_mask in zip(ac_code_ids+[end_id], mask_list):
                    if one_id not in one_mask:
                        # print(' '.join([self.vocabulary.id_to_word(i) for i in ac_code_ids]))
                        # print('id {} not in mask {}'.format(one_id, one_mask))
                        not_in_mask_error += 1
                        res = None
                        break
            except Exception as e:
                error_count += 1
                # print('slk error : {}'.format(str(e)))
                res = None
            return res

        if self.transformer is not None:
            print('before slk grammar: {}'.format(len(df)))
            df['grammar_mask_list'] = df['ac_code_word_id'].map(filter_slk)
            df = df[df['grammar_mask_list'].map(lambda x: x is not None)]
            print('after slk grammar: {}'.format(len(df)))

        return df

    def _get_raw_sample(self, row):
        # error_tokens = self.vocabulary.parse_text_without_pad([[k.value for k in self.data_df.iloc[index]["tokens"]]],
        #                                                       use_position_label=True)[0]
        # ac_tokens = self.vocabulary.parse_text_without_pad([[k.value for k in self.data_df.iloc[index]["ac_tokens"]]],
        #                                                       use_position_label=True)[0]
        sample = dict(row)
        sample['error_tokens'] = row['error_code_word_id']
        sample['error_tokens_name'] = row['error_code_word']
        sample['error_length'] = len(row['error_code_word_id'])
        sample['copy_length'] = len(row['error_code_word_id'])
        sample['includes'] = row['includes']
        if self.set_type != 'valid' and self.set_type != 'test' and self.set_type != 'deepfix':
            begin_id = self.vocabulary.word_to_id(self.vocabulary.begin_tokens[0])
            end_id = self.vocabulary.word_to_id(self.vocabulary.end_tokens[0])
            sample['ac_tokens'] = [begin_id] + row['ac_code_word_id'] + [end_id]
            sample['ac_length'] = len(sample['ac_tokens'])
            sample['pointer_map'] = [0] + row['pointer_map'] + [sample['error_length']-1]
            sample['is_copy'] = [0] + row['is_copy'] + [0]
            sample['distance'] = row['distance']
            if self.transformer is not None:
                sample['grammar_mask_list'] = self.transformer.get_all_token_mask_train([row['ac_code_word_id']])[0]
                sample['grammar_mask_length'] = [len(ma) for ma in sample['grammar_mask_list']]
            else:
                one_mask_list = sorted(set(row['ac_code_word_id'] + [end_id]) | self.keyword_ids)
                sample['grammar_mask_list'] = [one_mask_list for _ in range(sample['ac_length']-1)]
                mask_l = len(sample['grammar_mask_list'][0])
                sample['grammar_mask_length'] = [mask_l for _ in sample['grammar_mask_list']]
        else:
            sample['ac_tokens'] = None
            sample['ac_length'] = 0
            sample['pointer_map'] = None
            sample['is_copy'] = None
            sample['distance'] = 0
            sample['grammar_mask_list'] = None
            sample['grammar_mask_length'] = None
            # sample['grammar_mask_index_to_id_dict'] = None
            # sample['grammar_mask_id_to_index_dict'] = None

        if self.use_ast:
            code_graph = parse_ast_code_graph(sample['error_tokens_name'])
            sample['error_length'] = code_graph.graph_length
            in_seq, graph = code_graph.graph
            # begin_id = self.vocabulary.word_to_id(self.vocabulary.begin_tokens[0])
            # end_id = self.vocabulary.word_to_id(self.vocabulary.end_tokens[0])
            sample['error_tokens'] = [self.vocabulary.word_to_id(t) for t in in_seq]
            sample['adj'] = [[a, b] for a, b, _ in graph] + [[b, a] for a, b, _ in graph]
        return sample

    def add_samples(self, df):
        df = self.filter_df(df)
        self._samples += [row for i, row in df.iterrows()]

    def remain_samples(self, count=0, frac=1.0):
        if count != 0:
            self._samples = random.sample(self._samples, count)
        elif frac != 1:
            count = int(len(self._samples) * frac)
            self._samples = random.sample(self._samples, count)

    def combine_dataset(self, dataset):
        d = CCodeErrorDataSet(data_df=None, vocabulary=self.vocabulary, set_type=self.set_type, transform=self.transform,
                              transformer_vocab_slk=self.transformer)
        d._samples = self._samples + dataset._samples
        return d

    def remain_dataset(self, count=0, frac=1.0):
        d = CCodeErrorDataSet(data_df=None, vocabulary=self.vocabulary, set_type=self.set_type,
                              transform=self.transform, transformer_vocab_slk=self.transformer)
        d._samples = self._samples
        d.remain_samples(count=count, frac=frac)
        return d

    def __getitem__(self, index):
        return self._get_raw_sample(self._samples[index])

    def __len__(self):
        return len(self._samples)


def load_dataset(is_debug, vocabulary, mask_transformer, data_type=None, use_ast=False):
    if is_debug:
        data_dict = load_common_error_data_sample_100(data_type=data_type)
    else:
        data_dict = load_common_error_data(data_type=data_type)
    datasets = [CCodeErrorDataSet(pd.DataFrame(dd), vocabulary, name, transformer_vocab_slk=mask_transformer,
                                  use_ast=use_ast) for
                dd, name in
                zip(data_dict, ["train", "all_valid", "all_test"])]
    for d, n in zip(datasets, ["train", "val", "test"]):
        info_output = "There are {} parsed data in the {} dataset".format(len(d), n)
        print(info_output)
        # info(info_output)

    train_dataset, valid_dataset, test_dataset = datasets

    # ac_copy_data_dict = create_copy_addition_data(train_dataset.data_df['ac_code_word_id'],
    #                                               train_dataset.data_df['includes'])
    # ac_copy_dataset = CCodeErrorDataSet(pd.DataFrame(ac_copy_data_dict), vocabulary, 'ac_copy',
    #                                     transformer_vocab_slk=mask_transformer, no_filter=True)
    return train_dataset, valid_dataset, test_dataset, None


def load_deepfix_real_test_dataset():
    pass


def load_sample_save_dataset(is_debug, vocabulary, mask_transformer):
    if is_debug:
        data_dict = load_common_error_data_sample_100(addition_infomation=True)
    else:
        data_dict = load_common_error_data(addition_infomation=True)
    datasets = [CCodeErrorDataSet(pd.DataFrame(dd), vocabulary, name, transformer_vocab_slk=mask_transformer) for
                dd, name in
                zip(data_dict, ["train", "all_valid", "all_test"])]
    for d, n in zip(datasets, ["train", "val", "test"]):
        info_output = "There are {} parsed data in the {} dataset".format(len(d), n)
        print(info_output)
        # info(info_output)

    train_dataset, valid_dataset, test_dataset = datasets

    # ac_copy_data_dict = create_copy_addition_data(train_dataset.data_df['ac_code_word_id'],
    #                                               train_dataset.data_df['includes'])
    # ac_copy_dataset = CCodeErrorDataSet(pd.DataFrame(ac_copy_data_dict), vocabulary, 'ac_copy',
    #                                     transformer_vocab_slk=mask_transformer, no_filter=True)
    return train_dataset, valid_dataset, test_dataset, None


class DynamicFeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DynamicFeedForward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.weight = to_cuda(nn.Embedding(output_size, hidden_size))
        # self.bias = nn.Parameter(torch.rand(output_size))
        # self.bias = nn.Parameter(torch.autograd.Variable(torch.rand(output_size)))
        self.bias = to_cuda(torch.rand(output_size))
        self.relu = nn.ReLU()

    def forward(self, input_value, mask_tensor):
        """

        :param input_value: [batch, seq, hidden_size]
        :param mask_tensor: [batch, seq, mask_set]
        :param mask_mask: [batcg, seq, mask_set]
        :return:
        """
        combine_shape = list(input_value.shape[:-1])

        mask_tensor_shape = mask_tensor.shape
        # [batch, seq, mask, hidden]
        part_weight = self.weight(mask_tensor)
        # [batch, seq, mask]
        part_bias = torch.index_select(self.bias, dim=0, index=mask_tensor.view(-1)).view(mask_tensor_shape)

        # [-1, hidden, mask]
        part_weight = part_weight.view(-1, part_weight.shape[-2], part_weight.shape[-1]).permute(0, 2, 1)
        # [-1, mask]
        part_bias = part_bias.view(-1, part_bias.shape[-1])

        input_value = input_value.view(-1, 1, input_value.shape[-1])
        output_value = torch.bmm(input_value, part_weight).view(input_value.shape[0], mask_tensor_shape[-1]) + part_bias
        output_value = output_value.view(*combine_shape, mask_tensor_shape[-1])
        output_value = self.relu(output_value)
        return output_value


class IndexPositionEmbedding(nn.Module):
    def __init__(self, vocabulary_size, hidden_size, max_len):
        super(IndexPositionEmbedding, self).__init__()
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.position_embedding = nn.Embedding(max_len, hidden_size)
        self.embedding = nn.Embedding(vocabulary_size, hidden_size)

    def forward(self, inputs, input_mask=None, start_index=0):
        batch_size = inputs.shape[0]
        input_sequence_len = inputs.shape[1]
        embedded_input = self.embedding(inputs)
        position_input_index = to_cuda(
            torch.unsqueeze(torch.arange(start_index, start_index + input_sequence_len), dim=0).expand(batch_size, -1)).long()
        if input_mask is not None:
            position_input_index.masked_fill_(~input_mask, MAX_LENGTH)
        position_input_embedded = self.position_embedding(position_input_index)
        position_input = torch.cat([position_input_embedded, embedded_input], dim=-1)
        return position_input

    def forward_position(self, inputs, input_mask=None):
        batch_size = inputs.shape[0]
        input_sequence_len = inputs.shape[1]
        position_input_index = to_cuda(
            torch.unsqueeze(torch.arange(0, input_sequence_len), dim=0).expand(batch_size, -1)).long()
        if input_mask is not None:
            position_input_index.masked_fill_(~input_mask, MAX_LENGTH)
        position_input_embedded = self.position_embedding(position_input_index)
        return position_input_embedded

    def forward_content(self, inputs, input_mask=None):
        embedded_input = self.embedding(inputs)
        return embedded_input


class ContextEncoder(nn.Module):
    def __init__(self, rnn_type, hidden_size, n_layer, dropout_p):
        super(ContextEncoder, self).__init__()
        if rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=2 * hidden_size, hidden_size=hidden_size, num_layers=n_layer,
                              batch_first=True, bidirectional=True, dropout=dropout_p)

    def forward(self, seq, copy_length):
        seq_list = torch.unbind(seq, dim=0)
        copy_list = torch.unbind(copy_length, dim=0)
        token_seq = [s[:e] for s, e in zip(seq_list, copy_list)]
        hidden = self._encode(token_seq)
        return hidden

    def _encode(self, seq):
        packed_seq, idx_unsort = torch_util.pack_sequence(seq)
        # self.rnn.flatten_parameters()
        _, encoded_state = self.rnn(packed_seq)
        return torch.index_select(encoded_state, 1, idx_unsort.to(seq[0].device))





class RNNPointerNetworkModelWithSLKMask(nn.Module):
    def __init__(self, vocabulary_size, hidden_size, num_layers, start_label, end_label, graph_embedding, pointer_type,
                 graph_parameter, dropout_p=0.1, MAX_LENGTH=500,
                 atte_position_type='position', mask_transformer: TransformVocabularyAndSLK=None,
                 rnn_type='gru', rnn_layer_number=3, no_position_embedding=False, position_embedding_length=1000,
                 mask_type='static', vocabulary=None):
        super(RNNPointerNetworkModelWithSLKMask, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.start_label = start_label
        self.end_label = end_label
        self.dropout_p = dropout_p
        self.MAX_LENGTH = MAX_LENGTH
        self.atte_position_type = atte_position_type
        self.mask_type = mask_type

        if self.mask_type == 'static':
            self.keyword_ids_set = create_effect_keyword_ids_set(vocabulary)

        self.mask_transformer = mask_transformer
        self.graph_embedding = graph_embedding
        self.no_position_embedding = no_position_embedding

        if no_position_embedding:
            self.encode_position_embedding = nn.Embedding(vocabulary_size, hidden_size)
            self.decode_position_embedding = nn.Embedding(vocabulary_size, hidden_size)
            self.transform_encoder_embedding_to_atte = nn.Linear(hidden_size, hidden_size//2)
        else:
            self.encode_position_embedding = IndexPositionEmbedding(vocabulary_size, self.hidden_size//2, position_embedding_length)
            self.decode_position_embedding = IndexPositionEmbedding(vocabulary_size, self.hidden_size//2, position_embedding_length)

        self.encoder = EncoderRNN(vocab_size=vocabulary_size, max_len=MAX_LENGTH, input_size=self.hidden_size, hidden_size=self.hidden_size//2,
                                  n_layers=num_layers, bidirectional=True, rnn_cell=rnn_type,
                                  input_dropout_p=self.dropout_p, dropout_p=self.dropout_p, variable_lengths=False,
                                  embedding=None, update_embedding=True)

        self.graph_encoder = GraphEncoder(hidden_size=hidden_size, graph_embedding=graph_embedding,
                                          pointer_type=pointer_type, graph_parameter=graph_parameter,
                                          embedding=self.encode_position_embedding, embedding_size=hidden_size,
                                          p2_type='static', p2_step_length=1, do_embedding=False)

        self.context_encoder = ContextEncoder(rnn_type=rnn_type, hidden_size=hidden_size // 2, n_layer=rnn_layer_number,
                                          dropout_p=dropout_p)

        self.decoder = DecoderRNN(vocab_size=vocabulary_size, max_len=MAX_LENGTH, hidden_size=hidden_size,
                                  sos_id=start_label, eos_id=end_label, n_layers=num_layers, rnn_cell=rnn_type,
                                  bidirectional=False, input_dropout_p=self.dropout_p, dropout_p=self.dropout_p,
                                  use_attention=True)

        self.copy_linear = PositionWiseFeedForwardNet(input_size=self.hidden_size, hidden_size=self.hidden_size,
                                                      output_size=1, hidden_layer_count=1)

        # self.position_pointer = TrueMaskedMultiHeaderAttention(hidden_size=self.hidden_size, num_heads=1, attention_type='scaled_dot_product')
        # self.position_pointer = nn.Linear(self.hidden_size, self.hidden_size)
        self.position_pointer = PositionWiseFeedForwardNet(self.hidden_size, self.hidden_size, self.hidden_size//2, 1)
        # self.output_linear = nn.Linear(self.hidden_size, self.vocabulary_size)
        # self.output_linear = PositionWiseFeedForwardNet(input_size=self.hidden_size, hidden_size=self.hidden_size,
        #                                                 output_size=vocabulary_size, hidden_layer_count=1)

        self.dynamic_output_linear = DynamicFeedForward(input_size=self.hidden_size, hidden_size=self.hidden_size,
                                                output_size=vocabulary_size)

    def create_next_output(self, one_step_decoder_output, encoder_input_list, **kwargs):
        """

        :param copy_output: [batch, sequence, dim**]
        :param value_output: [batch, sequence, dim**, vocabulary_size]
        :param pointer_output: [batch, sequence, dim**, encode_length]
        :param encoder_input_list: [batch, encode_length, dim**]
        :return: [batch, sequence, dim**]
        """
        copy_output, value_output, pointer_output, slk_mask_tensor = one_step_decoder_output
        cant_pointer_pos = torch.eq(torch.sum(torch.ne(pointer_output, -float('inf')), dim=-1), 0)
        copy_output.masked_fill_(cant_pointer_pos, 0)
        pointer_output.masked_fill_(torch.unsqueeze(cant_pointer_pos, dim=-1), 0)
        is_copy = (copy_output > 0.5)
        _, top_id = torch.topk(F.softmax(value_output, dim=-1), k=1, dim=-1)
        # print('point_id: ', torch.sum(pointer_output))
        # print('top_id: ', torch.sum(top_id))
        top_id = torch.gather(slk_mask_tensor, dim=-1, index=top_id)
        _, pointer_pos_in_input = torch.topk(F.softmax(pointer_output, dim=-1), k=1, dim=-1)
        point_id = torch.gather(encoder_input_list, dim=-1, index=pointer_pos_in_input.squeeze(dim=-1))
        next_output = torch.where(is_copy, point_id, top_id.squeeze(dim=-1))
        # print('top_id: ', torch.sum(top_id))
        return next_output

    def decode_step(self, decode_input, encoder_hidden, decode_mask, encode_value, encode_mask, position_embedded,
                    teacher_forcing_ratio=1, value_mask=None, value_set=None, pointer_encoder_mask=None):
        # record_is_nan(encode_value, 'encode_value')
        decoder_output, hidden, _ = self.decoder(inputs=decode_input, encoder_hidden=encoder_hidden, encoder_outputs=encode_value,
                                            encoder_mask=~encode_mask, teacher_forcing_ratio=teacher_forcing_ratio)
        decoder_output = torch.stack(decoder_output, dim=1)
        # record_is_nan(decoder_output, 'decoder_output')
        if decode_mask is not None:
            decode_mask = decode_mask[:, 1:] if teacher_forcing_ratio == 1 else decode_mask
            decoder_output = decoder_output * decode_mask.view(decode_mask.shape[0], decode_mask.shape[1], *[1 for i in range(len(decoder_output.shape)-2)]).float()

        is_copy = F.sigmoid(self.copy_linear(decoder_output).squeeze(dim=-1))
        # record_is_nan(is_copy, 'is_copy in model: ')
        pointer_ff = self.position_pointer(decoder_output)
        # record_is_nan(pointer_ff, 'pointer_ff in model: ')
        pointer_output = torch.bmm(pointer_ff, torch.transpose(position_embedded, dim0=-1, dim1=-2))
        # record_is_nan(pointer_output, 'pointer_output in model: ')
        if pointer_encoder_mask is not None:
            dim_len = len(pointer_output.shape)
            pointer_output.masked_fill_(~pointer_encoder_mask.view(pointer_encoder_mask.shape[0], *[1 for i in range(dim_len-3)],
                                                                   pointer_encoder_mask.shape[-2], pointer_encoder_mask.shape[-1]),
                                        -float('inf'))
            if decode_mask is not None:
                pointer_output.masked_fill_(~decode_mask.view(decode_mask.shape[0], decode_mask.shape[1],
                                                                           *[1 for i in
                                                                             range(len(pointer_output.shape) - 2)]), 0)
            # pointer_output = torch.where(torch.unsqueeze(encode_mask, dim=1), pointer_output, to_cuda(torch.Tensor([float('-inf')])))

        # value_output = self.output_linear(decoder_output, value_set, value_mask)
        # print('before dynamic linear')
        # value_output = self.output_linear(decoder_output)
        value_output = self.dynamic_output_linear(decoder_output, value_set)
        # print('after dynamic linear')
        if value_mask is not None:
            dim_len = len(value_output.shape)
            value_output.masked_fill_(~value_mask.view(value_mask.shape[0], *[1 for i in range(dim_len - 3)],
                                                            value_mask.shape[-2], value_mask.shape[-1]), -float('inf'))
            if decode_mask is not None:
                value_output.masked_fill_(~decode_mask.view(decode_mask.shape[0], decode_mask.shape[1], *[1 for i in range(len(value_output.shape)-2)]), 0)
        return is_copy, value_output, pointer_output, hidden

    def forward(self, inputs, input_mask, output, output_mask, inputs_list, value_mask_set_tensor, mask_mask=None, pointer_encoder_mask=None,
                adjacent_matrix=None, copy_length=None, do_sample=False):
        if do_sample:
            return self.forward_dynamic_test(inputs, input_mask, inputs_list, adjacent_matrix, copy_length)

        # output_length = to_cuda(torch.sum(output_mask, dim=-1))
        # output_token_length = output_length - 2
        # print('vocabulary size: ', self.vocabulary_size)
        # python list [batch, seq, mask_list]
        # value_mask_set = self.mask_transformer.get_all_token_mask_train(ac_tokens_ids=output_list)
        # value_mask_set_len = [[len(ma) for ma in seq_mask] for seq_mask in value_mask_set]
        # mask_length = to_cuda(torch.LongTensor(PaddedList(value_mask_set_len))).view(-1)
        # mask_mask = create_sequence_length_mask(mask_length).view(output.shape[0], output.shape[1]-1, -1)
        # value_mask_set_tensor = to_cuda(torch.LongTensor(PaddedList(value_mask_set)))
        # print('value_mask_set shape : {}'.format(value_mask_set_tensor.shape))

        # print('after create tensor')
        # np_mask = np.array(paddded_mask)
        # value_mask = to_cuda(torch.ByteTensor(PaddedList(value_mask, shape=[output.shape[0], output.shape[1]-1, self.vocabulary_size])))
        # value_mask = None

        batch_size = inputs.shape[0]
        # position_input = self.encode_position_embedding(inputs, input_mask)
        if not self.no_position_embedding:
            position_input = self.encode_position_embedding(inputs, input_mask)
            if self.atte_position_type == 'position':
                encoder_atte_input_embedded = self.encode_position_embedding.forward_position(inputs, input_mask)
            elif self.atte_position_type == 'content':
                encoder_atte_input_embedded = self.encode_position_embedding.forward_content(inputs, input_mask)
            else:
                encoder_atte_input_embedded = self.encode_position_embedding.forward_content(inputs, input_mask)
        else:
            position_input = self.encode_position_embedding(inputs)
            encoder_atte_input_embedded = self.transform_encoder_embedding_to_atte(position_input)

        if self.graph_embedding is not None:
            p1_o, p2_o, encode_value = self.graph_encoder(adjacent_matrix, position_input, copy_length)
            encoder_hidden = self.context_encoder(encode_value, copy_length).permute(1, 0, 2)\
            .contiguous()\
            .view(batch_size, self.num_layers, -1)\
            .permute(1, 0, 2)\
            .contiguous()
        else:
            # position_input = self.encode_position_embedding(inputs, input_mask)
            encode_value, hidden = self.encoder(position_input)
            encoder_hidden = [hid.view(self.num_layers, hid.shape[1], -1) for hid in hidden]

        if self.no_position_embedding:
            position_output = self.decode_position_embedding(output)
        else:
            position_output = self.decode_position_embedding(inputs=output, input_mask=output_mask)
        is_copy, value_output, pointer_output, _ = self.decode_step(decode_input=position_output, encoder_hidden=encoder_hidden,
                                                                 decode_mask=output_mask, encode_value=encode_value,
                                                                 position_embedded=encoder_atte_input_embedded,
                                                                 encode_mask=input_mask, teacher_forcing_ratio=1,
                                                                    value_set=value_mask_set_tensor, value_mask=mask_mask,
                                                                    pointer_encoder_mask=pointer_encoder_mask)
        error_list = [0 for i in range(inputs.shape[0])]
        return is_copy, value_output, pointer_output, value_mask_set_tensor, error_list


    def forward_dynamic_test(self, inputs, input_mask, inputs_list, adjacent_matrix, copy_length):
        self.tmp_outputs_list = []
        batch_size = list(inputs.shape)[0]
        # position_input = self.encode_position_embedding(inputs, input_mask)
        if not self.no_position_embedding:
            position_input = self.encode_position_embedding(inputs, input_mask)
            if self.atte_position_type == 'position':
                encoder_atte_input_embedded = self.encode_position_embedding.forward_position(inputs, input_mask)
            elif self.atte_position_type == 'content':
                encoder_atte_input_embedded = self.encode_position_embedding.forward_content(inputs, input_mask)
            else:
                encoder_atte_input_embedded = self.encode_position_embedding.forward_content(inputs, input_mask)
        else:
            position_input = self.encode_position_embedding(inputs)
            encoder_atte_input_embedded = self.transform_encoder_embedding_to_atte(position_input)
        if self.graph_embedding is not None:
            p1_o, p2_o, encode_value = self.graph_encoder(adjacent_matrix, position_input, copy_length)
            hidden = self.context_encoder(encode_value, copy_length).permute(1, 0, 2) \
                .contiguous() \
                .view(batch_size, self.num_layers, -1) \
                .permute(1, 0, 2) \
                .contiguous()
        else:
            encode_value, hidden = self.encoder(position_input)
            hidden = [hid.view(self.num_layers, hid.shape[1], -1) for hid in hidden]

        if self.mask_type == 'slk':
            self.sample_decoder_list = [self.mask_transformer.create_new_slk_iterator() for i in range(batch_size)]
            self.special_set_list = [self.mask_transformer.create_id_constant_string_set_id_by_ids(code_ids)
                                     for code_ids in inputs_list]
        elif self.mask_type == 'static':
            self.compatible_tokens = None
            self.compatible_tokens_length = None


        dynamic_decoder = DynamicDecoder(start_label=self.start_label, end_label=self.end_label, pad_label=IGNORE_TOKEN,
                                         decoder_fn=self.decoder_one_step, create_next_output_fn=self.create_next_output,
                                         max_length=MAX_LENGTH)

        decoder_output_list, outputs_list, error_list = dynamic_decoder.decoder(encode_value, hidden, input_mask,
                                                                    encoder_input_list=inputs,
                                                                    encoder_atte_input_embedded=encoder_atte_input_embedded,
                                                                                encoder_inputs_python_list=inputs_list,
                                                                                copy_length=copy_length)

        is_copy, value_output, pointer_output, slk_mask = list(zip(*decoder_output_list))
        is_copy_result = torch.cat(is_copy, dim=1)
        padded_value_output = pad_last_dim_of_tensor_list(value_output, fill_value=-float('inf'))
        value_result = torch.cat(padded_value_output, dim=1)
        pointer_result = torch.cat(pointer_output, dim=1)
        padded_slk_mask = pad_last_dim_of_tensor_list(slk_mask, max_len=value_result.shape[-1])
        slk_mask_result = torch.cat(padded_slk_mask, dim=1)
        return is_copy_result, value_result, pointer_result, slk_mask_result, error_list

    def decoder_one_step(self, outputs, continue_mask, start_index, hidden,
                         encoder_output, encoder_atte_input_embedded, encoder_mask, **kwargs):

        self.tmp_outputs_list += [outputs]

        batch_size = outputs.shape[0]
        sequence_continue_mask = continue_mask.view(batch_size, 1)
        if self.no_position_embedding:
            output_embed = self.decode_position_embedding(outputs)
        else:
            output_embed = self.decode_position_embedding(inputs=outputs, input_mask=sequence_continue_mask,
                                                      start_index=start_index)
        continue_list = continue_mask.tolist()
        outputs_list = outputs.view(-1).tolist()
        # out_word = [self.mask_transformer.vocab.id_to_word(out) for out in outputs_list]
        # print(out_word)
        # print('continue: ', continue_list)

        if self.mask_type == 'slk':
            error_list, mask_mask, pointer_mask_tensor, slk_mask_tensor = self.create_slk_one_step_token_masks(batch_size,
                                                                                                               continue_list,
                                                                                                               encoder_output,
                                                                                                               kwargs,
                                                                                                               outputs_list,
                                                                                                               start_index)
        elif self.mask_type == 'static':
            copy_length = kwargs['copy_length']
            ori_input_seq = kwargs['encoder_input_list']
            slk_mask_tensor, mask_length = self.create_one_step_token_masks(ori_input_seq, continue_mask, copy_length=copy_length)
            mask_mask = create_sequence_length_mask(mask_length.view(-1)).view(batch_size, 1, -1)
            error_list = []
            pointer_list = [[1 for _ in range(c)] for c in copy_length.tolist()]
            pointer_tensor = torch.ByteTensor(PaddedList(pointer_list, shape=[batch_size, encoder_output.shape[1]])).to(encoder_output.device)
            pointer_mask_tensor = torch.unsqueeze(pointer_tensor, dim=1)

        is_copy, value_output, pointer_output, hidden = self.decode_step(decode_input=output_embed,
                                                                         encoder_hidden=hidden,
                                                                         decode_mask=sequence_continue_mask,
                                                                         encode_value=encoder_output,
                                                                         position_embedded=encoder_atte_input_embedded,
                                                                         encode_mask=encoder_mask,
                                                                         teacher_forcing_ratio=0,
                                                                         value_set=slk_mask_tensor,
                                                                         value_mask=mask_mask,
                                                                         pointer_encoder_mask=pointer_mask_tensor)
        return (is_copy, value_output, pointer_output, slk_mask_tensor), hidden, error_list

    def create_slk_one_step_token_masks(self, batch_size, continue_list, encoder_output,
                                        kwargs, outputs_list, start_index):

        def get_candicate_one_step_fn(t_parser, output_id, previous_id_set_list):
            try:
                token_list = self.mask_transformer.get_candicate_step(t_parser, output_id, previous_id_set_list=previous_id_set_list)
                token_list = [token_list]
                # res = [self.mask_transformer.get_candicate_step(t_parser, output_id, previous_id_set_list=previous_id_set_list)]
            except Exception as e:
                res = None
                token_list = None
                output_ids_list = torch.cat(self.tmp_outputs_list, dim=1).tolist()
                vocab = self.mask_transformer.vocab
                output_words_list = [[vocab.id_to_word(o) for o in b] for b in output_ids_list]
                for code in output_words_list:
                    c = ' '.join(code)
                    # print(c)
                    info(c)
                # raise Exception(str(e))
            return token_list

        if start_index != 0:
            slk_mask_list = [
                get_candicate_one_step_fn(t_parser, output_id, previous_id_set_list=special_set)
                if con else [OrderedList({self.mask_transformer.vocab.word_to_id(self.end_label)})]
                for t_parser, output_id, special_set, con in
                zip(self.sample_decoder_list, outputs_list, self.special_set_list, continue_list)]

        else:
            slk_mask_list = [
                get_candicate_one_step_fn(t_parser, None, previous_id_set_list=special_set)
                if con else [OrderedList({self.mask_transformer.vocab.word_to_id(self.end_label)})]
                for t_parser, output_id, special_set, con in
                zip(self.sample_decoder_list, outputs_list, self.special_set_list, continue_list)]
        # error_list = [1 if mask_list is None else 0 for mask_list in slk_mask_list]
        # slk_mask_list = [[0] if mask_list is None else mask_list for mask_list in slk_mask_list]
        error_list = []
        for i, mask_list in enumerate(slk_mask_list):
            if mask_list is None:
                error_list += [i]
                slk_mask_list[i] = [[0]]
        encoder_input_list = kwargs['encoder_inputs_python_list']
        pointer_mask_tensor = self.create_one_step_pointer_mask(encoder_output, encoder_input_list, slk_mask_list)
        len_list = [[len(one_mask) for one_mask in masks] for masks in slk_mask_list]
        max_mask_len = max(more_itertools.collapse(len_list))
        slk_mask_tensor = to_cuda(torch.Tensor(PaddedList(slk_mask_list, shape=(batch_size, 1, max_mask_len))).long())
        mask_length = to_cuda(torch.LongTensor(PaddedList(len_list, shape=[batch_size, 1])))
        mask_mask = create_sequence_length_mask(mask_length.view(-1)).view(batch_size, 1, -1)
        return error_list, mask_mask, pointer_mask_tensor, slk_mask_tensor

    def create_one_step_pointer_mask(self, encoder_output, encoder_input_list, slk_mask_list):
        batch_size = encoder_output.shape[0]
        pointer_mask = [[[tok in slk_mask for tok in inp_seq] for slk_mask in slk_mask_seq]
                        for inp_seq, slk_mask_seq in zip(encoder_input_list, slk_mask_list)]
        pointer_mask_tensor = to_cuda(
            torch.ByteTensor(PaddedList(pointer_mask, shape=[batch_size, 1, encoder_output.shape[1]])))
        return pointer_mask_tensor

    def create_one_step_token_masks(self, ori_input_seq, continue_mask, copy_length=None):
        if not hasattr(self, 'compatible_tokens') or self.compatible_tokens is None:
            self.compatible_tokens, self.compatible_tokens_length = self.create_static_token_mask(ori_input_seq,
                                                                                                  copy_length)
        return self.compatible_tokens, self.compatible_tokens_length

    def create_static_token_mask(self, ori_input_seq, copy_length):
        ori_input_seq_list = ori_input_seq.tolist()
        copy_length_list = copy_length.tolist()
        mask_list = []
        for inp, c in zip(ori_input_seq_list, copy_length_list):
            input_set = self.keyword_ids_set | set(inp[:c]) | {self.end_label}
            mask_list += [[sorted(input_set)]]
        mask_len_list = [[len(j) for j in i] for i in mask_list]
        compatible_tokens = torch.LongTensor(PaddedList(mask_list)).to(ori_input_seq.device)
        compatible_tokens_length = torch.LongTensor(PaddedList(mask_len_list)).to(ori_input_seq.device)
        return compatible_tokens, compatible_tokens_length


def create_parse_rnn_input_batch_data_fn(vocab, use_ast=False):

    def parse_rnn_input_batch_data(batch_data, do_sample=False, add_value_mask=False):
        error_tokens = to_cuda(torch.LongTensor(PaddedList(batch_data['error_tokens'])))
        max_input = max(batch_data['error_length'])
        input_mask = to_cuda(create_sequence_length_mask(to_cuda(torch.LongTensor(batch_data['error_length'])), max_len=max_input))
        copy_length = to_cuda(torch.LongTensor(batch_data['copy_length']))

        if not do_sample:
            # ac_tokens_decoder_input = [bat[:-1] for bat in batch_data['ac_tokens']]
            ac_tokens_decoder_input = batch_data['ac_tokens']
            ac_tokens = to_cuda(torch.LongTensor(PaddedList(ac_tokens_decoder_input)))
            ac_tokens_length = [len(bat) for bat in ac_tokens_decoder_input]
            max_output = max(ac_tokens_length)
            output_mask = to_cuda(create_sequence_length_mask(to_cuda(torch.LongTensor(ac_tokens_length)), max_len=max_output))
        else:
            ac_tokens = None
            output_mask = None

        if add_value_mask:
            token_sets = [create_token_set(one, vocab) for one in batch_data['error_tokens']]
            if not do_sample:
                ac_token_sets = [create_token_set(one, vocab) for one in batch_data['ac_tokens']]
                token_sets = [tok_s | ac_tok_s for tok_s, ac_tok_s in zip(token_sets, ac_token_sets)]
            token_mask = [create_token_mask_by_token_set(one, vocab.vocabulary_size) for one in token_sets]
            token_mask_tensor = to_cuda(torch.ByteTensor(token_mask))
        else:
            token_mask_tensor = None

        batch_size = len(batch_data['error_tokens'])

        # value_ac_tokens = batch_data['ac_tokens']
        # value_ac_tokens = [tokens[1:-1] for tokens in value_ac_tokens]
        # value_mask_list = transformer.get_all_token_mask_train(value_ac_tokens)
        # value_mask_set_len = [[len(ma) for ma in mask_list] for mask_list in value_mask_list]
        # batch_data['grammar_mask_list'] = value_mask_list

        value_mask_list = batch_data['grammar_mask_list']
        value_mask_set_len = batch_data['grammar_mask_length']
        max_seq = max([len(s) for s in value_mask_list])
        max_mask_len = max(more_itertools.collapse(value_mask_set_len))
        value_mask_set_tensor = to_cuda(torch.LongTensor(PaddedList(value_mask_list, shape=[batch_size, max_seq, max_mask_len])))
        # [batch, seq, mask_len]
        mask_length = to_cuda(torch.LongTensor(PaddedList(value_mask_set_len, shape=[batch_size, max_seq])))
        mask_length_shape = list(mask_length.shape)
        mask_mask = create_sequence_length_mask(mask_length.view(-1)).view(mask_length_shape[0], mask_length_shape[1], -1)

        if not do_sample:
            pointer_encoder_mask = [[[inp in mask for inp in inp_seq] if cpy else [1 for i in range(len(inp_seq))]for cpy, mask in zip(copy_seq[1:], mask_seq)]
                                    for inp_seq, copy_seq, mask_seq in
                                    zip(batch_data['error_tokens'], batch_data['is_copy'], batch_data['grammar_mask_list'])]
            pointer_encoder_mask_tensor = to_cuda(torch.ByteTensor(PaddedList(pointer_encoder_mask, shape=[batch_size, max_seq, max_input])))
        else:
            pointer_encoder_mask_tensor = None

        if use_ast:
            adjacent_tuple = [[[i] + tt for tt in t] for i, t in enumerate(batch_data['adj'])]
            adjacent_tuple = [list(t) for t in unzip(more_itertools.flatten(adjacent_tuple))]
            size = max(batch_data['error_length'])
            # print("max length in this batch:{}".format(size))
            adjacent_tuple = torch.LongTensor(adjacent_tuple)
            adjacent_values = torch.ones(adjacent_tuple.shape[1]).long()
            adjacent_size = torch.Size([len(batch_data['error_length']), size, size])
            info('batch_data input_length: ' + str(batch_data['error_length']))
            info('size: ' + str(size))
            info('adjacent_tuple: ' + str(adjacent_tuple.shape))
            info('adjacent_size: ' + str(adjacent_size))
            adjacent_matrix = to_cuda(
                torch.sparse.LongTensor(
                    adjacent_tuple,
                    adjacent_values,
                    adjacent_size,
                ).float().to_dense()
            )
        else:
            adjacent_matrix = None

        # return error_tokens, input_mask, ac_tokens, output_mask
        return error_tokens, input_mask, ac_tokens, output_mask, batch_data['error_tokens'], \
               value_mask_set_tensor, mask_mask, pointer_encoder_mask_tensor, \
               adjacent_matrix, copy_length
    return parse_rnn_input_batch_data


def create_parse_target_batch_data():

    def parse_target_batch_data(batch_data):
        ac_tokens_decoder_output = [bat[1:] for bat in batch_data['ac_tokens']]
        target_ac_tokens = to_cuda(torch.LongTensor(PaddedList(ac_tokens_decoder_output, fill_value=IGNORE_TOKEN)))
        grammar_id_to_index_dict = [[{m: i for i, m in enumerate(masks)} for masks in grammar_mask_list]
                                    for grammar_mask_list in batch_data['grammar_mask_list']]
        ac_tokens_value_output = [[di[tok] for tok, di in zip(ac_tokens, value_mask_dict)]
                                    for ac_tokens, value_mask_dict in
                                    zip(ac_tokens_decoder_output, grammar_id_to_index_dict)]
        target_value_tokens = to_cuda(torch.LongTensor(PaddedList(ac_tokens_value_output, fill_value=IGNORE_TOKEN)))

        ac_tokens_length = [len(bat) for bat in ac_tokens_decoder_output]
        max_output = max(ac_tokens_length)
        output_mask = to_cuda(create_sequence_length_mask(to_cuda(torch.LongTensor(ac_tokens_length)), max_len=max_output))

        pointer_map = [bat[1:] for bat in batch_data['pointer_map']]
        target_pointer_output = to_cuda(torch.LongTensor(PaddedList(pointer_map, fill_value=IGNORE_TOKEN)))

        is_copy = [bat[1:] for bat in batch_data['is_copy']]
        target_is_copy = to_cuda(torch.Tensor(PaddedList(is_copy)))

        return target_is_copy, target_pointer_output, target_value_tokens, target_ac_tokens, output_mask
    return parse_target_batch_data

def create_combine_loss_fn(copy_weight=1, pointer_weight=1, value_weight=1, average_value=True):
    copy_loss_fn = nn.BCELoss(reduce=False)
    pointer_loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_TOKEN, reduce=False)
    value_loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_TOKEN, reduce=False)

    def combine_total_loss(is_copy, value_log_probs, pointer_log_probs, slk_mask_result, error_list,
                           target_copy, target_pointer, target_value,
                           target_ac_token, output_mask):
        # record_is_nan(is_copy, 'is_copy: ')
        # record_is_nan(pointer_log_probs, 'pointer_log_probs: ')
        # record_is_nan(value_log_probs, 'value_log_probs: ')
        output_mask_float = output_mask.float()
        # info('output_mask_float: {}, mask batch: {}'.format(str(output_mask_float), torch.sum(output_mask_float, dim=-1)))
        # info('target_copy: {}, mask batch: {}'.format(str(target_copy), torch.sum(target_copy, dim=-1)))


        no_copy_weight = torch.where(target_copy.byte(), target_copy, to_cuda(torch.Tensor([15]).view(*[1 for i in range(len(target_copy.shape))])))
        copy_loss = copy_loss_fn(is_copy, target_copy)
        # record_is_nan(copy_loss, 'copy_loss: ')
        copy_loss = copy_loss * output_mask_float * no_copy_weight
        # record_is_nan(copy_loss, 'copy_loss with mask: ')

        pointer_log_probs = permute_last_dim_to_second(pointer_log_probs)
        pointer_loss = pointer_loss_fn(pointer_log_probs, target_pointer)
        # record_is_nan(pointer_loss, 'pointer_loss: ')
        pointer_loss = pointer_loss * output_mask_float
        # record_is_nan(pointer_loss, 'pointer_loss with mask: ')

        value_log_probs = permute_last_dim_to_second(value_log_probs)
        value_loss = value_loss_fn(value_log_probs, target_value)
        # record_is_nan(value_loss, 'value_loss: ')
        value_mask_float = output_mask_float * (~target_copy.byte()).float()
        # info('value_mask_float: {}, mask batch: {}'.format(str(value_mask_float), torch.sum(value_mask_float, dim=-1)))
        value_loss = value_loss * value_mask_float
        # record_is_nan(value_loss, 'value_loss with mask: ')
        total_count = torch.sum(output_mask_float)
        if average_value:
            value_count = torch.sum(value_mask_float)
            value_loss = value_loss / value_count * total_count

        total_loss = copy_weight*copy_loss + pointer_weight*pointer_loss + value_weight*value_loss
        # total_loss = pointer_weight*pointer_loss
        return torch.sum(total_loss) / total_count
    return combine_total_loss


def create_output_ids(model_output, model_input, do_sample=False):
    """

    :param copy_output: [batch, sequence, dim**]
    :param value_output: [batch, sequence, dim**, vocabulary_size]
    :param pointer_output: [batch, sequence, dim**, encode_length]
    :param input: [batch, encode_length, dim**]
    :return: [batch, sequence, dim**]
    """
    is_copy, value_output, pointer_output, mask_tensor, error_list = model_output
    error_tokens = model_input[0]
    is_copy = (is_copy > 0.5)
    _, top_id = torch.topk(F.softmax(value_output, dim=-1), k=1, dim=-1)
    top_id = torch.gather(mask_tensor, dim=-1, index=top_id)
    _, pointer_pos_in_input = torch.topk(F.softmax(pointer_output, dim=-1), k=1, dim=-1)
    pointer_pos_in_input = pointer_pos_in_input.squeeze(dim=-1)
    point_id = torch.gather(error_tokens, dim=-1, index=pointer_pos_in_input)
    next_output = torch.where(is_copy, point_id, top_id.squeeze(dim=-1))
    return next_output


def slk_expand_output_and_target(model_output, model_target):
    model_output = list(model_output)
    model_target = list(model_target)
    model_output[0], model_target[0] = expand_tensor_sequence_to_same(model_output[0], model_target[0], fill_value=0)
    model_output[1], model_target[1] = expand_tensor_sequence_to_same(model_output[1], model_target[1], fill_value=IGNORE_TOKEN)
    model_output[2], model_target[2] = expand_tensor_sequence_to_same(model_output[2], model_target[2], fill_value=IGNORE_TOKEN)

    expand_len = model_output[2].shape[1]
    model_output[3] = expand_tensor_sequence_len(model_output[3], max_len=expand_len, fill_value=0)
    model_target[3] = expand_tensor_sequence_len(model_target[3], max_len=expand_len, fill_value=IGNORE_TOKEN)
    model_target[4] = expand_tensor_sequence_len(model_target[4], max_len=expand_len, fill_value=0)
    return model_output, model_target


def create_save_sample_data(vocab):
    start_id = vocab.word_to_id(vocab.begin_tokens[0])
    end_id = vocab.word_to_id(vocab.end_tokens[0])
    unk_id = vocab.word_to_id(vocab.unk)
    def add_model_data_record(output_ids, model_output, batch_data):
        saved_list = []
        error_list = model_output[4]
        output_ids_list = output_ids.tolist()
        for i, output_ids in enumerate(output_ids_list):
            if error_list[i] == 1:
                continue
            sample_code, end_pos = convert_one_token_ids_to_code(output_ids, vocab.id_to_word, start_id, end_id, unk_id)
            if end_pos is None:
                continue
            rec = [batch_data['id'][i], batch_data['user_id'][i], batch_data['problem_id'][i],
                   batch_data['problem_user_id'][i], '\n'.join(batch_data['includes'][i]), batch_data['code'][i],
                   sample_code, batch_data['similar_code'][i], batch_data['original_modify_action_list'][i],
                   batch_data['original_distance'][i]]
            saved_list += [rec]
        return saved_list
    return add_model_data_record


def create_multi_step_next_outputs_fn():
    def create_multi_step_next_outputs():
        pass
    return create_multi_step_next_outputs
