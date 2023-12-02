import os
import random

import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

import config
from common import torch_util
from common.args_util import to_cuda, get_model
from common.logger import init_a_file_logger, info
from common.opt import OpenAIAdam
from common.torch_util import create_sequence_length_mask, permute_last_dim_to_second, expand_tensor_sequence_len
from common.util import data_loader, show_process_map, PaddedList, convert_one_token_ids_to_code, filter_token_ids, \
    compile_c_code_by_gcc, create_token_set, create_token_mask_by_token_set
from experiment.experiment_util import load_common_error_data, load_common_error_data_sample_100, \
    create_addition_error_data, create_copy_addition_data, load_deepfix_error_data
from experiment.parse_xy_util import parse_output_and_position_map
from model.base_attention import TransformEncoderModel, TransformDecoderModel, TrueMaskedMultiHeaderAttention, \
    register_hook, PositionWiseFeedForwardNet, is_nan, record_is_nan
from read_data.load_data_vocabulary import create_common_error_vocabulary
from seq2seq.models import EncoderRNN, DecoderRNN
from vocabulary.word_vocabulary import Vocabulary
from common.constants import pre_defined_c_tokens, pre_defined_c_library_tokens


MAX_LENGTH = 500
IGNORE_TOKEN = -1
is_debug = False


class CCodeErrorDataSet(Dataset):
    def __init__(self,
                 data_df: pd.DataFrame,
                 vocabulary: Vocabulary,
                 set_type: str,
                 transform=None):
        self.set_type = set_type
        self.transform = transform
        self.vocabulary = vocabulary
        if data_df is not None:
            self.data_df = self.filter_df(data_df)
            self._samples = [self._get_raw_sample(row) for i, row in self.data_df.iterrows()]
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
        return df

    def _get_raw_sample(self, row):
        # error_tokens = self.vocabulary.parse_text_without_pad([[k.value for k in self.data_df.iloc[index]["tokens"]]],
        #                                                       use_position_label=True)[0]
        # ac_tokens = self.vocabulary.parse_text_without_pad([[k.value for k in self.data_df.iloc[index]["ac_tokens"]]],
        #                                                       use_position_label=True)[0]

        sample = {"error_tokens": row['error_code_word_id'],
                  'error_length': len(row['error_code_word_id']),
                  'includes': row['includes']}
        if self.set_type != 'valid' and self.set_type != 'test' and self.set_type != 'deepfix':
            begin_id = self.vocabulary.word_to_id(self.vocabulary.begin_tokens[0])
            end_id = self.vocabulary.word_to_id(self.vocabulary.end_tokens[0])
            sample['ac_tokens'] = [begin_id] + row['ac_code_word_id'] + [end_id]
            sample['ac_length'] = len(sample['ac_tokens'])
            # sample['token_map'] = self.data_df.iloc[index]['token_map']
            # sample['pointer_map'] = create_pointer_map(sample['ac_length'], sample['token_map'])
            sample['pointer_map'] = [0] + row['pointer_map'] + [sample['error_length']-1]
            # sample['error_mask'] = self.data_df.iloc[index]['error_mask']
            sample['is_copy'] = [0] + row['is_copy'] + [0]
            sample['distance'] = row['distance']
        else:
            sample['ac_tokens'] = None
            sample['ac_length'] = 0
            # sample['token_map'] = None
            sample['pointer_map'] = None
            # sample['error_mask'] = None
            sample['is_copy'] = None
            sample['distance'] = 0
        return sample

    def add_samples(self, df):
        df = self.filter_df(df)
        self._samples += [self._get_raw_sample(row) for i, row in df.iterrows()]

    def remain_samples(self, count=0, frac=1.0):
        if count != 0:
            self._samples = random.sample(self._samples, count)
        elif frac != 1:
            count = int(len(self._samples) * frac)
            self._samples = random.sample(self._samples, count)

    def combine_dataset(self, dataset):
        d = CCodeErrorDataSet(data_df=None, vocabulary=self.vocabulary, set_type=self.set_type, transform=self.transform)
        d._samples = self._samples + dataset._samples
        return d

    def remain_dataset(self, count=0, frac=1.0):
        d = CCodeErrorDataSet(data_df=None, vocabulary=self.vocabulary, set_type=self.set_type,
                              transform=self.transform)
        d._samples = self._samples
        d.remain_samples(count=count, frac=frac)
        return d

    def __getitem__(self, index):
        return self._samples[index]

    def __len__(self):
        return len(self._samples)


def create_pointer_map(ac_length, token_map):
    """
    the ac length includes begin and end label.
    :param ac_length:
    :param token_map:
    :return: map ac id to error pointer position with begin and end
    """
    pointer_map = [-1 for i in range(ac_length)]
    for error_i, ac_i in enumerate(token_map):
        if ac_i >= 0:
            pointer_map[ac_i + 1] = error_i
    last_point = -1
    for i in range(len(pointer_map)):
        if pointer_map[i] == -1:
            pointer_map[i] = last_point + 1
        else:
            last_point = pointer_map[i]
            if last_point + 1 >= len(token_map):
                last_point = len(token_map) - 2
    return pointer_map


# class SinCosPositionEmbeddingModel(nn.Module):
#     def __init__(self, min_timescale=1.0, max_timescale=1.0e4):
#         super(SinCosPositionEmbeddingModel, self).__init__()
#         self.min_timescale = min_timescale
#         self.max_timescale = max_timescale
#
#     def forward(self, x, position_start_list=None):
#         """
#
#         :param x: has more than 3 dims
#         :param position_start_list: len(position_start_list) == len(x.shape) - 2. default: [0] * num_dims.
#                 create position from start to start+length-1 for each dim.
#         :param min_timescale:
#         :param max_timescale:
#         :return:
#         """
#         x_shape = list(x.shape)
#         num_dims = len(x_shape) - 2
#         channels = x_shape[-1]
#         num_timescales = channels // (num_dims * 2)
#         log_timescales_increment = (math.log(float(self.max_timescale) / float(self.min_timescale)) / (float(num_timescales) - 1))
#         inv_timescales = self.min_timescale * to_cuda(torch.exp(torch.range(0, num_timescales - 1) * -log_timescales_increment))
#         # add moved position start index
#         if position_start_list is None:
#             position_start_list = [0] * num_dims
#         for dim in range(num_dims):
#             length = x_shape[dim + 1]
#             # position = transform_to_cuda(torch.range(0, length-1))
#             # create position from start to start+length-1 for each dim
#             position = to_cuda(torch.range(position_start_list[dim], position_start_list[dim] + length - 1))
#             scaled_time = torch.unsqueeze(position, 1) * torch.unsqueeze(inv_timescales, 0)
#             signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
#             prepad = dim * 2 * num_timescales
#             postpad = channels - (dim + 1) * 2 * num_timescales
#             signal = F.pad(signal, (prepad, postpad, 0, 0))
#             for _ in range(dim + 1):
#                 signal = torch.unsqueeze(signal, dim=0)
#             for _ in range(num_dims - dim - 1):
#                 signal = torch.unsqueeze(signal, dim=-2)
#             x += signal
#         return x


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

# class SelfAttentionPointerNetworkModel(nn.Module):
#     def __init__(self, vocabulary_size, hidden_size, encoder_stack_num, decoder_stack_num, start_label, end_label, dropout_p=0.1, num_heads=2, normalize_type=None, MAX_LENGTH=500):
#         super(SelfAttentionPointerNetworkModel, self).__init__()
#         self.vocabulary_size = vocabulary_size
#         self.hidden_size = hidden_size
#         self.encoder_stack_num = encoder_stack_num
#         self.decoder_stack_num = decoder_stack_num
#         self.start_label = start_label
#         self.end_label = end_label
#         self.dropout_p = dropout_p
#         self.num_heads = num_heads
#         self.normalize_type = normalize_type
#         self.MAX_LENGTH = MAX_LENGTH
#
#         # self.encode_embedding = nn.Embedding(vocabulary_size, hidden_size//2)
#         # self.encode_position_embedding = SinCosPositionEmbeddingModel()
#         self.encode_position_embedding = IndexPositionEmbedding(vocabulary_size, self.hidden_size//2, MAX_LENGTH+1)
#         # self.decode_embedding = nn.Embedding(vocabulary_size, hidden_size//2)
#         # self.decode_position_embedding = SinCosPositionEmbeddingModel()
#         self.decode_position_embedding = IndexPositionEmbedding(vocabulary_size, self.hidden_size//2, MAX_LENGTH+1)
#
#         self.encoder = TransformEncoderModel(hidden_size=self.hidden_size, encoder_stack_num=self.encoder_stack_num, dropout_p=self.dropout_p, num_heads=self.num_heads, normalize_type=self.normalize_type)
#         self.decoder = TransformDecoderModel(hidden_size=hidden_size, decoder_stack_num=self.decoder_stack_num, dropout_p=self.dropout_p, num_heads=self.num_heads, normalize_type=self.normalize_type)
#
#         # self.copy_linear = nn.Linear(self.hidden_size, 1)
#         self.copy_linear = PositionWiseFeedForwardNet(input_size=self.hidden_size, hidden_size=self.hidden_size,
#                                                       output_size=1, hidden_layer_count=1)
#
#         # self.position_pointer = TrueMaskedMultiHeaderAttention(hidden_size=self.hidden_size, num_heads=1, attention_type='scaled_dot_product')
#         # self.position_pointer = nn.Linear(self.hidden_size, self.hidden_size)
#         self.position_pointer = PositionWiseFeedForwardNet(self.hidden_size, self.hidden_size, self.hidden_size//2, 1)
#         # self.output_linear = nn.Linear(self.hidden_size, self.vocabulary_size)
#         self.output_linear = PositionWiseFeedForwardNet(input_size=self.hidden_size, hidden_size=self.hidden_size,
#                                                       output_size=vocabulary_size, hidden_layer_count=1)
#
#     def create_next_output(self, copy_output, value_output, pointer_output, input):
#         """
#
#         :param copy_output: [batch, sequence, dim**]
#         :param value_output: [batch, sequence, dim**, vocabulary_size]
#         :param pointer_output: [batch, sequence, dim**, encode_length]
#         :param input: [batch, encode_length, dim**]
#         :return: [batch, sequence, dim**]
#         """
#         is_copy = (copy_output > 0.5)
#         _, top_id = torch.topk(F.softmax(value_output, dim=-1), k=1, dim=-1)
#         _, pointer_pos_in_input = torch.topk(F.softmax(pointer_output, dim=-1), k=1, dim=-1)
#         point_id = torch.gather(input, dim=-1, index=pointer_pos_in_input.squeeze(dim=-1))
#         next_output = torch.where(is_copy, point_id, top_id.squeeze(dim=-1))
#         return next_output
#
#     def decode_step(self, decode_input, decode_mask, encode_value, encode_mask, position_embedded):
#         # record_is_nan(encode_value, 'encode_value')
#         decoder_output = self.decoder(decode_input, decode_mask, encode_value, encode_mask)
#         # record_is_nan(decoder_output, 'decoder_output')
#
#         is_copy = F.sigmoid(self.copy_linear(decoder_output).squeeze(dim=-1))
#         # record_is_nan(is_copy, 'is_copy in model: ')
#         pointer_ff = self.position_pointer(decoder_output)
#         # record_is_nan(pointer_ff, 'pointer_ff in model: ')
#         pointer_output = torch.bmm(pointer_ff, torch.transpose(position_embedded, dim0=-1, dim1=-2))
#         # record_is_nan(pointer_output, 'pointer_output in model: ')
#         if encode_mask is not None:
#             dim_len = len(pointer_output.shape)
#             pointer_output.masked_fill_(~encode_mask.view(encode_mask.shape[0], *[1 for i in range(dim_len-2)], encode_mask.shape[-1]), -float('inf'))
#             # pointer_output = torch.where(torch.unsqueeze(encode_mask, dim=1), pointer_output, to_cuda(torch.Tensor([float('-inf')])))
#
#         value_output = self.output_linear(decoder_output)
#         return is_copy, value_output, pointer_output
#
#     def forward(self, input, input_mask, output, output_mask):
#         position_input = self.encode_position_embedding(input, input_mask)
#         position_input_embedded = self.encode_position_embedding.forward_position(input, input_mask)
#         encode_value = self.encoder(position_input, input_mask)
#
#         position_output = self.decode_position_embedding(output, output_mask)
#         is_copy, value_output, pointer_output = self.decode_step(position_output, output_mask, encode_value, input_mask,
#                                                                  position_embedded=position_input_embedded)
#         return is_copy, value_output, pointer_output

class RNNPointerNetworkModel(nn.Module):
    def __init__(self, vocabulary_size, hidden_size, num_layers, start_label, end_label, dropout_p=0.1, MAX_LENGTH=500, atte_position_type='position'):
        super(RNNPointerNetworkModel, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.start_label = start_label
        self.end_label = end_label
        self.dropout_p = dropout_p
        self.MAX_LENGTH = MAX_LENGTH
        self.atte_position_type = atte_position_type

        self.encode_position_embedding = IndexPositionEmbedding(vocabulary_size, self.hidden_size//2, MAX_LENGTH+1)
        self.decode_position_embedding = IndexPositionEmbedding(vocabulary_size, self.hidden_size//2, MAX_LENGTH+1)

        self.encoder = EncoderRNN(vocab_size=vocabulary_size, max_len=MAX_LENGTH, input_size=self.hidden_size, hidden_size=self.hidden_size//2,
                                  n_layers=num_layers, bidirectional=True, rnn_cell='lstm',
                                  input_dropout_p=self.dropout_p, dropout_p=self.dropout_p, variable_lengths=False,
                                  embedding=None, update_embedding=True)
        self.decoder = DecoderRNN(vocab_size=vocabulary_size, max_len=MAX_LENGTH, hidden_size=hidden_size,
                                  sos_id=start_label, eos_id=end_label, n_layers=num_layers, rnn_cell='lstm',
                                  bidirectional=False, input_dropout_p=self.dropout_p, dropout_p=self.dropout_p,
                                  use_attention=True)

        self.copy_linear = PositionWiseFeedForwardNet(input_size=self.hidden_size, hidden_size=self.hidden_size,
                                                      output_size=1, hidden_layer_count=1)

        # self.position_pointer = TrueMaskedMultiHeaderAttention(hidden_size=self.hidden_size, num_heads=1, attention_type='scaled_dot_product')
        # self.position_pointer = nn.Linear(self.hidden_size, self.hidden_size)
        self.position_pointer = PositionWiseFeedForwardNet(self.hidden_size, self.hidden_size, self.hidden_size//2, 1)
        # self.output_linear = nn.Linear(self.hidden_size, self.vocabulary_size)
        self.output_linear = PositionWiseFeedForwardNet(input_size=self.hidden_size, hidden_size=self.hidden_size,
                                                      output_size=vocabulary_size, hidden_layer_count=1)

    def create_next_output(self, copy_output, value_output, pointer_output, input):
        """

        :param copy_output: [batch, sequence, dim**]
        :param value_output: [batch, sequence, dim**, vocabulary_size]
        :param pointer_output: [batch, sequence, dim**, encode_length]
        :param input: [batch, encode_length, dim**]
        :return: [batch, sequence, dim**]
        """
        is_copy = (copy_output > 0.5)
        _, top_id = torch.topk(F.softmax(value_output, dim=-1), k=1, dim=-1)
        _, pointer_pos_in_input = torch.topk(F.softmax(pointer_output, dim=-1), k=1, dim=-1)
        point_id = torch.gather(input, dim=-1, index=pointer_pos_in_input.squeeze(dim=-1))
        next_output = torch.where(is_copy, point_id, top_id.squeeze(dim=-1))
        return next_output

    def decode_step(self, decode_input, encoder_hidden, decode_mask, encode_value, encode_mask, position_embedded, teacher_forcing_ratio=1, value_mask=None):
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
        if encode_mask is not None:
            dim_len = len(pointer_output.shape)
            pointer_output.masked_fill_(~encode_mask.view(encode_mask.shape[0], *[1 for i in range(dim_len-2)], encode_mask.shape[-1]), -float('inf'))
            # pointer_output = torch.where(torch.unsqueeze(encode_mask, dim=1), pointer_output, to_cuda(torch.Tensor([float('-inf')])))

        value_output = self.output_linear(decoder_output)
        if value_mask is not None:
            dim_len = len(value_output.shape)
            value_output.masked_fill_(~value_mask.view(value_mask.shape[0], *[1 for i in range(dim_len - 2)], value_mask.shape[-1]), -float('inf'))
        return is_copy, value_output, pointer_output, hidden

    def forward(self, inputs, input_mask, output, output_mask, value_mask=None, test=False):
        if test:
            return self.forward_test(inputs, input_mask, value_mask=value_mask)
        position_input = self.encode_position_embedding(inputs, input_mask)
        if self.atte_position_type == 'position':
            encoder_atte_input_embedded = self.encode_position_embedding.forward_position(inputs, input_mask)
        elif self.atte_position_type == 'content':
            encoder_atte_input_embedded = self.encode_position_embedding.forward_content(inputs, input_mask)
        else:
            encoder_atte_input_embedded = self.encode_position_embedding.forward_position(inputs, input_mask)
        encode_value, hidden = self.encoder(position_input)
        encoder_hidden = [hid.view(self.num_layers, hid.shape[1], -1) for hid in hidden]

        position_output = self.decode_position_embedding(inputs=output, input_mask=output_mask)
        is_copy, value_output, pointer_output, _ = self.decode_step(decode_input=position_output, encoder_hidden=encoder_hidden,
                                                                 decode_mask=output_mask, encode_value=encode_value,
                                                                 position_embedded=encoder_atte_input_embedded,
                                                                 encode_mask=input_mask, teacher_forcing_ratio=1,
                                                                    value_mask=value_mask)
        return is_copy, value_output, pointer_output

    def forward_test(self, input, input_mask, value_mask):
        position_input = self.encode_position_embedding(input, input_mask)
        if self.atte_position_type == 'position':
            encoder_atte_input_embedded = self.encode_position_embedding.forward_position(input, input_mask)
        elif self.atte_position_type == 'content':
            encoder_atte_input_embedded = self.encode_position_embedding.forward_content(input, input_mask)
        else:
            encoder_atte_input_embedded = self.encode_position_embedding.forward_position(input, input_mask)
        encode_value, hidden = self.encoder(position_input)
        hidden = [hid.view(self.num_layers, hid.shape[1], -1) for hid in hidden]

        batch_size = list(input.shape)[0]
        continue_mask = to_cuda(torch.Tensor([1 for i in range(batch_size)])).byte()
        outputs = to_cuda(torch.LongTensor([[self.start_label] for i in range(batch_size)]))
        is_copy_stack = []
        value_output_stack = []
        pointer_stack = []
        output_stack = []

        for i in range(self.MAX_LENGTH):
            cur_input_mask = continue_mask.view(continue_mask.shape[0], 1)
            output_embed = self.decode_position_embedding(inputs=outputs, input_mask=cur_input_mask, start_index=i)

            is_copy, value_output, pointer_output, hidden = self.decode_step(decode_input=output_embed, encoder_hidden=hidden,
                                                                 decode_mask=cur_input_mask, encode_value=encode_value,
                                                                 position_embedded=encoder_atte_input_embedded,
                                                                 encode_mask=input_mask, teacher_forcing_ratio=0,
                                                                             value_mask=value_mask)
            is_copy_stack.append(is_copy)
            value_output_stack.append(value_output)
            pointer_stack.append(pointer_output)

            outputs = self.create_next_output(is_copy, value_output, pointer_output, input)
            output_stack.append(outputs)

            cur_continue = torch.ne(outputs, self.end_label).view(outputs.shape[0])
            continue_mask = continue_mask & cur_continue

            if torch.sum(continue_mask) == 0:
                break

        is_copy_result = torch.cat(is_copy_stack, dim=1)
        value_result = torch.cat(value_output_stack, dim=1)
        pointer_result = torch.cat(pointer_stack, dim=1)
        output_result = torch.cat(output_stack, dim=1)
        return output_result, is_copy_result, value_result, pointer_result


def parse_input_batch_data(batch_data):
    error_tokens = to_cuda(torch.LongTensor(PaddedList(batch_data['error_tokens'])))
    max_input = max(batch_data['error_length'])
    input_mask = to_cuda(create_sequence_length_mask(to_cuda(torch.LongTensor(batch_data['error_length'])), max_len=max_input))

    ac_tokens_decoder_input = [bat[:-1] for bat in batch_data['ac_tokens']]
    ac_tokens = to_cuda(torch.LongTensor(PaddedList(ac_tokens_decoder_input)))
    ac_tokens_length = [len(bat) for bat in ac_tokens_decoder_input]
    max_output = max(ac_tokens_length)
    output_mask = to_cuda(create_sequence_length_mask(to_cuda(torch.LongTensor(ac_tokens_length)), max_len=max_output))

    # info('error_tokens shape: {}, input_mask shape: {}, ac_tokens shape: {}, output_mask shape: {}'.format(
    #     error_tokens.shape, input_mask.shape, ac_tokens.shape, output_mask.shape))
    # info('error_length: {}'.format(batch_data['error_length']))
    # info('ac_length: {}'.format(batch_data['ac_length']))

    return error_tokens, input_mask, ac_tokens, output_mask


def parse_rnn_input_batch_data(batch_data, vocab, do_sample=False, add_value_mask=False):
    error_tokens = to_cuda(torch.LongTensor(PaddedList(batch_data['error_tokens'])))
    max_input = max(batch_data['error_length'])
    input_mask = to_cuda(create_sequence_length_mask(to_cuda(torch.LongTensor(batch_data['error_length'])), max_len=max_input))

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

    # info('error_tokens shape: {}, input_mask shape: {}, ac_tokens shape: {}, output_mask shape: {}'.format(
    #     error_tokens.shape, input_mask.shape, ac_tokens.shape, output_mask.shape))
    # info('error_length: {}'.format(batch_data['error_length']))
    # info('ac_length: {}'.format(batch_data['ac_length']))

    # return error_tokens, input_mask, ac_tokens, output_mask
    return error_tokens, input_mask, ac_tokens, output_mask, token_mask_tensor


def parse_target_batch_data(batch_data):
    ac_tokens_decoder_output = [bat[1:] for bat in batch_data['ac_tokens']]
    target_ac_tokens = to_cuda(torch.LongTensor(PaddedList(ac_tokens_decoder_output, fill_value=IGNORE_TOKEN)))

    ac_tokens_length = [len(bat) for bat in ac_tokens_decoder_output]
    max_output = max(ac_tokens_length)
    output_mask = to_cuda(create_sequence_length_mask(to_cuda(torch.LongTensor(ac_tokens_length)), max_len=max_output))

    pointer_map = [bat[1:] for bat in batch_data['pointer_map']]
    target_pointer_output = to_cuda(torch.LongTensor(PaddedList(pointer_map, fill_value=IGNORE_TOKEN)))

    is_copy = [bat[1:] for bat in batch_data['is_copy']]
    target_is_copy = to_cuda(torch.Tensor(PaddedList(is_copy)))

    return target_is_copy, target_pointer_output, target_ac_tokens, output_mask

def create_combine_loss_fn(copy_weight=1, pointer_weight=1, value_weight=1, average_value=True):
    copy_loss_fn = nn.BCELoss(reduce=False)
    pointer_loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_TOKEN, reduce=False)
    value_loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_TOKEN, reduce=False)

    def combine_total_loss(is_copy, pointer_log_probs, value_log_probs, target_copy, target_pointer, target_value, output_mask):
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


def create_output_ids(is_copy, value_output, pointer_output, error_tokens):
    """

    :param copy_output: [batch, sequence, dim**]
    :param value_output: [batch, sequence, dim**, vocabulary_size]
    :param pointer_output: [batch, sequence, dim**, encode_length]
    :param input: [batch, encode_length, dim**]
    :return: [batch, sequence, dim**]
    """
    is_copy = (is_copy > 0.5)
    _, top_id = torch.topk(F.softmax(value_output, dim=-1), k=1, dim=-1)
    _, pointer_pos_in_input = torch.topk(F.softmax(pointer_output, dim=-1), k=1, dim=-1)
    pointer_pos_in_input = pointer_pos_in_input.squeeze(dim=-1)
    point_id = torch.gather(error_tokens, dim=-1, index=pointer_pos_in_input)
    next_output = torch.where(is_copy, point_id, top_id.squeeze(dim=-1))
    return next_output


def train(model, dataset, batch_size, loss_function, optimizer, clip_norm, epoch_ratio, vocab, add_value_mask=False):
    total_loss = to_cuda(torch.Tensor([0]))
    count = to_cuda(torch.Tensor([0]))
    steps = 0
    total_accuracy = to_cuda(torch.Tensor([0]))
    model.train()

    with tqdm(total=(len(dataset)*epoch_ratio)) as pbar:
        for batch_data in data_loader(dataset, batch_size=batch_size, is_shuffle=True, drop_last=True, epoch_ratio=epoch_ratio):
            model.zero_grad()
            # target_ac_tokens = to_cuda(torch.LongTensor(PaddedList(batch_data['ac_tokens'])))
            # target_pointer_output = to_cuda(torch.LongTensor(PaddedList(batch_data['pointer_map'], fill_value=IGNORE_TOKEN)))
            # target_is_copy = to_cuda(torch.Tensor(PaddedList(batch_data['is_copy'])))
            # max_output = max(batch_data['ac_length'])

            # model_input = parse_input_batch_data(batch_data)
            model_input = parse_rnn_input_batch_data(batch_data, vocab, do_sample=False, add_value_mask=add_value_mask)
            is_copy, value_output, pointer_output = model.forward(*model_input)
            if steps % 100 == 0:
                pointer_output_id = torch.squeeze(torch.topk(F.softmax(pointer_output, dim=-1), k=1, dim=-1)[1], dim=-1)
                info(pointer_output_id)
                # print(pointer_output_id)

            model_target = parse_target_batch_data(batch_data)
            target_is_copy, target_pointer_output, target_ac_tokens, output_mask = model_target
            loss = loss_function(is_copy, pointer_output, value_output, *model_target)
            # record_is_nan(loss, 'total loss in train:')


            # if steps == 0:
            #     pointer_probs = F.softmax(pointer_output, dim=-1)
                # print('pointer output softmax: ', pointer_probs)
                # print('pointer output: ', torch.squeeze(torch.topk(pointer_probs, k=1, dim=-1)[1], dim=-1))
                # print('target_pointer_output: ', target_pointer_output)

            batch_count = torch.sum(output_mask).float()
            batch_loss = loss * batch_count
            # loss = batch_loss / batch_count

            # if clip_norm is not None:
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            loss.backward()
            optimizer.step()

            output_ids = create_output_ids(is_copy, value_output, pointer_output, model_input[0])
            batch_accuracy = torch.sum(torch.eq(output_ids, target_ac_tokens) & output_mask).float()
            accuracy = batch_accuracy / batch_count
            total_accuracy += batch_accuracy
            total_loss += batch_loss

            step_output = 'in train step {}  loss: {}, accracy: {}'.format(steps, loss.data.item(), accuracy.data.item())
            # print(step_output)
            info(step_output)

            count += batch_count
            steps += 1
            pbar.update(batch_size)

    return (total_loss/count).item(), (total_accuracy/count).item()


def evaluate(model, dataset, batch_size, loss_function, vocab, do_sample=False, print_output=False, add_value_mask=False):
    total_loss = to_cuda(torch.Tensor([0]))
    count = to_cuda(torch.Tensor([0]))
    total_batch = to_cuda(torch.Tensor([0]))
    steps = 0
    total_accuracy = to_cuda(torch.Tensor([0]))
    total_correct = to_cuda(torch.Tensor([0]))
    model.eval()
    start_id = vocab.word_to_id(vocab.begin_tokens[0])
    end_id = vocab.word_to_id(vocab.end_tokens[0])
    unk_id = vocab.word_to_id(vocab.unk)

    with tqdm(total=len(dataset)) as pbar:
        with torch.no_grad():
            for batch_data in data_loader(dataset, batch_size=batch_size, drop_last=True):
                model.zero_grad()
                # target_ac_tokens = to_cuda(torch.LongTensor(PaddedList(batch_data['ac_tokens'])))
                # target_pointer_output = to_cuda(torch.LongTensor(PaddedList(batch_data['pointer_map'])))
                # target_is_copy = to_cuda(torch.Tensor(PaddedList(batch_data['is_copy'])))
                # max_output = max(batch_data['ac_length'])
                # output_mask = create_sequence_length_mask(to_cuda(torch.LongTensor(batch_data['ac_length'])),
                #                                           max_len=max_output)


                # model_input = parse_input_batch_data(batch_data)
                model_input = parse_rnn_input_batch_data(batch_data, vocab, do_sample=do_sample, add_value_mask=add_value_mask)
                model_target = parse_target_batch_data(batch_data)
                target_is_copy, target_pointer_output, target_ac_tokens, output_mask = model_target
                # is_copy, value_output, pointer_output = model.forward(*model_input)
                if do_sample:
                    _, is_copy, value_output, pointer_output = model.forward(*model_input, do_sample=True)
                    predict_len = is_copy.shape[1]
                    target_len = target_is_copy.shape[1]
                    expand_len = max(predict_len, target_len)
                    is_copy = expand_tensor_sequence_len(is_copy, max_len=expand_len, fill_value=0)
                    value_output = expand_tensor_sequence_len(value_output, max_len=expand_len, fill_value=0)
                    pointer_output = expand_tensor_sequence_len(pointer_output, max_len=expand_len, fill_value=0)
                    target_is_copy = expand_tensor_sequence_len(target_is_copy, max_len=expand_len, fill_value=0)
                    target_pointer_output = expand_tensor_sequence_len(target_pointer_output, max_len=expand_len, fill_value=IGNORE_TOKEN)
                    target_ac_tokens = expand_tensor_sequence_len(target_ac_tokens, max_len=expand_len, fill_value=IGNORE_TOKEN)
                    output_mask = expand_tensor_sequence_len(output_mask, max_len=expand_len, fill_value=0)
                else:
                    is_copy, value_output, pointer_output = model.forward(*model_input)


                loss = loss_function(is_copy, pointer_output, value_output, target_is_copy, target_pointer_output,
                                     target_ac_tokens, output_mask)
                # if steps == 0:
                #     pointer_probs = F.softmax(pointer_output, dim=-1)
                    # print('pointer output softmax: ', pointer_probs)
                    # print('pointer output: ', torch.squeeze(torch.topk(pointer_probs, k=1, dim=-1)[1], dim=-1))
                    # print('target_pointer_output: ', target_pointer_output)

                batch_count = torch.sum(output_mask).float()
                batch_loss = loss * batch_count
                # loss = batch_loss / batch_count

                output_ids = create_output_ids(is_copy, value_output, pointer_output, model_input[0])
                batch_accuracy = torch.sum(torch.eq(output_ids, target_ac_tokens) & output_mask).float()
                batch_correct = torch.sum(
                    torch.eq(torch.sum(torch.ne(output_ids, target_ac_tokens) & output_mask, dim=-1), 0)).float()
                correct = batch_correct / batch_size
                accuracy = batch_accuracy / batch_count
                total_accuracy += batch_accuracy
                total_correct += batch_correct
                total_loss += batch_loss
                total_batch += batch_size

                step_output = 'in evaluate step {}  loss: {}, accracy: {}, correct: {}'.format(steps, loss.data.item(), accuracy.data.item(), correct.data.item())
                # print(step_output)
                info(step_output)

                if print_output and steps % 10 == 0:
                    output_ids = output_ids.tolist()
                    target_ids = batch_data['ac_tokens']
                    is_copy = (is_copy > 0.5).tolist()
                    target_is_copy = target_is_copy.tolist()
                    value_output = torch.squeeze(torch.topk(F.softmax(value_output, dim=-1), k=1, dim=-1)[1], dim=-1)
                    value_output = value_output.tolist()
                    target_ac_tokens = target_ac_tokens.tolist()
                    pointer_output = torch.squeeze(torch.topk(F.softmax(pointer_output, dim=-1), k=1, dim=-1)[1], dim=-1)
                    pointer_output = pointer_output.tolist()
                    target_pointer_output = target_pointer_output.tolist()
                    target_length = torch.sum(output_mask, dim=-1)
                    target_length = target_length.tolist()
                    for out, tar, cop, tar_cop, val, tar_val, poi, tar_poi, tar_len in zip(output_ids, target_ids, is_copy,
                                                                                  target_is_copy, value_output,
                                                                                  target_ac_tokens,
                                                                                  pointer_output,
                                                                                  target_pointer_output, target_length):
                    # for out, tar,  in zip(output_ids, target_ids):
                        out_code, end_pos = convert_one_token_ids_to_code(out, id_to_word_fn=vocab.id_to_word, start=start_id,
                                                             end=end_id, unk=unk_id)
                        tar_code, tar_end_pos = convert_one_token_ids_to_code(tar[1:], id_to_word_fn=vocab.id_to_word, start=start_id,
                                                             end=end_id, unk=unk_id)
                        info('-------------- step {} ------------------------'.format(steps))
                        info('output: {}'.format(out_code))
                        info('target: {}'.format(tar_code))
                        cop = [str(c) for c in cop]
                        tar_cop = [str(int(c)) for c in tar_cop]
                        poi = [str(c) for c in poi]
                        tar_poi = [str(c) for c in tar_poi]
                        info('copy output: {}'.format(' '.join(cop[:tar_len])))
                        info('copy target: {}'.format(' '.join(tar_cop[:tar_len])))
                        info('pointer output: {}'.format(' '.join(poi[:tar_len])))
                        info('pointer target: {}'.format(' '.join(tar_poi[:tar_len])))

                        value_list = []
                        target_list = []
                        for c, v, t in zip(tar_cop, val, tar_val):
                            if c == '1':
                                value_list += ['<COPY>']
                                target_list += ['<COPY>']
                            else:
                                value_list += [vocab.id_to_word(int(v))]
                                target_list += [vocab.id_to_word(int(t))]
                        info('value output: {}'.format(' '.join(value_list[:tar_len])))
                        info('value target: {}'.format(' '.join(target_list[:tar_len])))


                count += batch_count
                steps += 1
                pbar.update(batch_size)

    return (total_loss / count).item(), (total_accuracy / count).item(), (total_correct/total_batch).item()


def evaluate_multi_step(model, dataset, batch_size, vocab, do_sample=False, print_output=False, max_sample_times=1,
                        file_path=None, no_target=False, drop_last=True, add_value_mask=False):
    count = to_cuda(torch.Tensor([0]))
    total_batch = to_cuda(torch.Tensor([0]))
    steps = 0
    total_accuracy = to_cuda(torch.Tensor([0]))
    total_correct = to_cuda(torch.Tensor([0]))
    total_compile_success = 0.0
    total_first_compile_success = 0.0
    model.eval()
    start_id = vocab.word_to_id(vocab.begin_tokens[0])
    end_id = vocab.word_to_id(vocab.end_tokens[0])
    unk_id = vocab.word_to_id(vocab.unk)
    max_sample_times = 1 if max_sample_times < 1 else max_sample_times

    with tqdm(total=len(dataset)) as pbar:
        with torch.no_grad():
            for batch_data in data_loader(dataset, batch_size=batch_size, drop_last=drop_last):
                model.zero_grad()
                # target_ac_tokens = to_cuda(torch.LongTensor(PaddedList(batch_data['ac_tokens'])))
                # target_pointer_output = to_cuda(torch.LongTensor(PaddedList(batch_data['pointer_map'])))
                # target_is_copy = to_cuda(torch.Tensor(PaddedList(batch_data['is_copy'])))
                # max_output = max(batch_data['ac_length'])
                # output_mask = create_sequence_length_mask(to_cuda(torch.LongTensor(batch_data['ac_length'])),
                #                                           max_len=max_output)
                cur_batch_size = len(batch_data['error_tokens'])
                batch_continue = [True for i in range(cur_batch_size)]
                batch_compile_success = [False for i in range(cur_batch_size)]
                total_output_records = []
                total_compile_records = []
                total_continue_records = []
                total_is_copy_records = []
                total_pointer_records = []
                total_value_records = []
                first_sample_success = 0

                input_data = batch_data.copy()
                target_data = batch_data.copy()

                for t in range(max_sample_times):
                    # model_input = parse_input_batch_data(batch_data)
                    model_input = parse_rnn_input_batch_data(input_data, vocab=vocab, do_sample=do_sample, add_value_mask=add_value_mask)
                    # error_tokens, input_mask, ac_tokens, output_mask = model_input
                    # is_copy, value_output, pointer_output = model.forward(*model_input)
                    if do_sample:
                        _, is_copy, value_output, pointer_output = model.forward(*model_input, do_sample=True)
                    else:
                        is_copy, value_output, pointer_output = model.forward(*model_input)
                    output_ids = create_output_ids(is_copy, value_output, pointer_output, model_input[0])
                    output_ids_list = output_ids.tolist()
                    is_copy = (is_copy > 0.5)
                    _, top_id = torch.topk(F.softmax(value_output, dim=-1), k=1, dim=-1)
                    _, pointer_pos_in_input = torch.topk(F.softmax(pointer_output, dim=-1), k=1, dim=-1)
                    pointer_pos_in_input = pointer_pos_in_input.squeeze(dim=-1)
                    top_id = top_id.squeeze(dim=-1)
                    is_copy_list = is_copy.tolist()
                    pointer_pos_in_input = pointer_pos_in_input.tolist()
                    top_id = top_id.tolist()
                    total_is_copy_records += [is_copy_list]
                    total_pointer_records += [pointer_pos_in_input]
                    total_value_records += [top_id]

                    compile_result = []
                    cur_continue = []
                    cur_end_pos = []
                    # compile the continue batch output and record
                    for out, inc, bat_con in zip(output_ids_list, batch_data['includes'], batch_continue):
                        res = False
                        con = False
                        end_pos = 1
                        if bat_con:
                            out_code, end_pos = convert_one_token_ids_to_code(out, id_to_word_fn=vocab.id_to_word, start=start_id,
                                                                          end=end_id, unk=unk_id, includes=inc)
                            if end_pos is not None:
                                res = compile_c_code_by_gcc(out_code, file_path)
                                con = not res
                        compile_result += [res]
                        cur_continue += [con]
                        cur_end_pos += [end_pos]
                    if t == 0:
                        success_l = [1 if c else 0 for c in compile_result]
                        first_sample_success = sum(success_l)
                        total_first_compile_success += first_sample_success

                    next_input_id = []
                    # create next input token ids
                    for bat_con, out, err in zip(batch_continue, output_ids_list, input_data['error_tokens']):
                        if bat_con:
                            ids, error_pos = filter_token_ids(out, start=start_id, end=end_id, unk=unk_id)
                        else:
                            ids = err
                        next_input_id += [ids]
                    input_data['error_tokens'] = next_input_id
                    input_data['error_length'] = [len(toks) for toks in input_data['error_tokens']]
                    total_output_records += [input_data['error_tokens']]

                    batch_continue = [b and c for b, c in zip(batch_continue, cur_continue)]
                    batch_compile_success = [b or c for b, c in zip(batch_compile_success, compile_result)]
                    total_continue_records += [batch_continue]
                    total_compile_records += [batch_compile_success]

                    tmp_max_len = is_copy.shape[1]
                    cur_end_pos = [pos - 1 if pos is not None else tmp_max_len for pos in cur_end_pos]
                    cur_end_pos_tensor = to_cuda(torch.LongTensor(cur_end_pos))
                    end_pos_mask = create_sequence_length_mask(cur_end_pos_tensor, max_len=tmp_max_len)
                    if torch.sum(~is_copy & end_pos_mask) == 0:
                        break

                output_ids = to_cuda(torch.LongTensor(PaddedList(input_data['error_tokens'])))
                if not no_target:
                    model_target = parse_target_batch_data(target_data)
                    target_is_copy, target_pointer_output, target_ac_tokens, output_mask = model_target
                    batch_count = torch.sum(output_mask).float()

                    if do_sample:
                        predict_len = output_ids.shape[1]
                        target_len = target_ac_tokens.shape[1]
                        expand_len = max(predict_len, target_len)
                        output_ids = expand_tensor_sequence_len(output_ids, max_len=expand_len, fill_value=0)
                        target_ac_tokens = expand_tensor_sequence_len(target_ac_tokens, max_len=expand_len,
                                                                      fill_value=IGNORE_TOKEN)
                        output_mask = expand_tensor_sequence_len(output_mask, max_len=expand_len, fill_value=0)

                    batch_accuracy = torch.sum(torch.eq(output_ids, target_ac_tokens) & output_mask).float()
                    batch_correct = torch.sum(
                        torch.eq(torch.sum(torch.ne(output_ids, target_ac_tokens) & output_mask, dim=-1), 0)).float()
                    correct = (batch_correct / cur_batch_size).data.item()
                    accuracy = (batch_accuracy / batch_count).data.item()
                    total_accuracy += batch_accuracy
                    total_correct += batch_correct

                    target_is_copy_list = target_is_copy.long().tolist()
                    target_pointer_output_list = target_pointer_output.tolist()
                else:
                    correct = 0
                    accuracy = 0
                    batch_count = 0

                compile_success_count = 0
                for r in batch_compile_success:
                    if r:
                        compile_success_count += 1
                total_compile_success += compile_success_count

                step_output = '|      in evaluate step {}, accracy: {}, correct: {}, compile: {}, ' \
                              'first_compile: {}'.format(steps, accuracy, correct, compile_success_count/cur_batch_size,
                                                         first_sample_success/cur_batch_size)
                # print(step_output)
                info(step_output)

                if print_output and steps % 10 == 0:
                    tmp_count = 0
                    for out_list, con_list, res_list, err, is_copy_list, pointer_list, value_list in \
                            zip(list(zip(*total_output_records)), list(zip(*total_continue_records)),
                                list(zip(*total_compile_records)), batch_data['error_tokens'],
                                list(zip(*total_is_copy_records)), list(zip(*total_pointer_records)),
                                list(zip(*total_value_records)),):
                        info('------------------------------- in step {} record {} -------------------------------'
                             .format(steps, tmp_count))
                        # print input
                        err_code, err_end_pos = convert_one_token_ids_to_code(err, id_to_word_fn=vocab.id_to_word,
                                                                              start=start_id,
                                                                              end=end_id, unk=unk_id)
                        info('input : {}'.format(err_code))
                        if not no_target:
                            tar_is_copy = target_is_copy_list[tmp_count]
                            tar_pointer = target_pointer_output_list[tmp_count]
                            tar = input_data['ac_tokens'][tmp_count]

                            tar_code, tar_end_pos = convert_one_token_ids_to_code(tar[1:], id_to_word_fn=vocab.id_to_word,
                                                                                  start=start_id,
                                                                                  end=end_id, unk=unk_id)
                            tar_len = len(tar[1:])
                            tar_val = [str(vocab.id_to_word(v)) if c == 0 else '<COPY>' for c, v in
                                   zip(tar_is_copy[:tar_len], tar[1:1+tar_len])]

                        s = 0
                        for out, con, res, is_cop, poi, val in zip(out_list, con_list, res_list, is_copy_list, pointer_list, value_list):
                            out_code, end_pos = convert_one_token_ids_to_code(out, id_to_word_fn=vocab.id_to_word,
                                                                              start=start_id,
                                                                              end=end_id, unk=unk_id)
                            out_len = len(out)
                            info('--------------------------- step {} {} th records {} times iteration------------------------'
                                 .format(steps, tmp_count, s))
                            info('continue: {}, compile result: {}'.format(con, res))
                            info('output: {}'.format(out_code))
                            if not no_target:
                                info('target: {}'.format(tar_code))
                            is_cop_str = [str(is_c) for is_c in is_cop[:out_len+1]]
                            info('output is_copy: {}'.format(' '.join(is_cop_str)))
                            if not no_target:
                                tar_is_cop_str = [str(is_c) for is_c in tar_is_copy[:tar_len]]
                                info('target is_copy: {}'.format(' '.join(tar_is_cop_str)))
                            poi_str = [str(is_c) for is_c in poi[:out_len+1]]
                            info('output pointer: {}'.format(' '.join(poi_str)))
                            if not no_target:
                                tar_poi_str = [str(is_c) for is_c in tar_pointer[:tar_len]]
                                info('target pointer: {}'.format(' '.join(tar_poi_str)))


                            val = [str(vocab.id_to_word(v)) if c == 0 else '<COPY>' for c, v in zip(is_cop[:out_len+1], val[:out_len+1])]

                            info('output value: {}'.format(' '.join(val)))
                            if not no_target:
                                info('target value: {}'.format(' '.join(tar_val)))
                            s += 1
                        tmp_count += 1

                count += batch_count
                steps += 1
                total_batch += cur_batch_size
                pbar.update(cur_batch_size)

        # output_result, is_copy, value_output, pointer_output = model.forward_test(*parse_test_batch_data(batch_data))

    return (total_compile_success / total_batch).item(), (total_accuracy / count).item(), (total_correct/total_batch).item(), (total_first_compile_success / total_batch).item()


def sample_better_output(model, dataset, batch_size, vocab, do_sample=False, epoch_ratio=1.0, add_value_mask=False):
    steps = 0
    model.eval()
    start_id = vocab.word_to_id(vocab.begin_tokens[0])
    end_id = vocab.word_to_id(vocab.end_tokens[0])
    unk_id = vocab.word_to_id(vocab.unk)

    records = []

    with tqdm(total=(len(dataset) * epoch_ratio)) as pbar:
        with torch.no_grad():
            for batch_data in data_loader(dataset, batch_size=batch_size, drop_last=True, epoch_ratio=epoch_ratio):
                model.zero_grad()
                # target_ac_tokens = to_cuda(torch.LongTensor(PaddedList(batch_data['ac_tokens'])))
                # target_pointer_output = to_cuda(torch.LongTensor(PaddedList(batch_data['pointer_map'])))
                # target_is_copy = to_cuda(torch.Tensor(PaddedList(batch_data['is_copy'])))
                # max_output = max(batch_data['ac_length'])
                # output_mask = create_sequence_length_mask(to_cuda(torch.LongTensor(batch_data['ac_length'])),
                #                                           max_len=max_output)

                # model_input = parse_input_batch_data(batch_data)
                model_input = parse_rnn_input_batch_data(batch_data, vocab=vocab, do_sample=do_sample, add_value_mask=add_value_mask)
                model_target = parse_target_batch_data(batch_data)
                target_is_copy, target_pointer_output, target_ac_tokens, output_mask = model_target
                # is_copy, value_output, pointer_output = model.forward(*model_input)
                if do_sample:
                    _, is_copy, value_output, pointer_output = model.forward(*model_input, do_sample=True)
                    predict_len = is_copy.shape[1]
                    target_len = target_is_copy.shape[1]
                    expand_len = max(predict_len, target_len)
                    is_copy = expand_tensor_sequence_len(is_copy, max_len=expand_len, fill_value=0)
                    value_output = expand_tensor_sequence_len(value_output, max_len=expand_len, fill_value=0)
                    pointer_output = expand_tensor_sequence_len(pointer_output, max_len=expand_len, fill_value=0)
                    # target_is_copy = expand_tensor_sequence_len(target_is_copy, max_len=expand_len, fill_value=0)
                    # target_pointer_output = expand_tensor_sequence_len(target_pointer_output, max_len=expand_len,
                    #                                                    fill_value=IGNORE_TOKEN)
                    target_ac_tokens = expand_tensor_sequence_len(target_ac_tokens, max_len=expand_len,
                                                                  fill_value=IGNORE_TOKEN)
                    # output_mask = expand_tensor_sequence_len(output_mask, max_len=expand_len, fill_value=0)
                else:
                    is_copy, value_output, pointer_output = model.forward(*model_input)

                output_ids = create_output_ids(is_copy, value_output, pointer_output, model_input[0])

                output_ids = output_ids.tolist()
                target_ac_tokens = target_ac_tokens.tolist()
                for out, tar, dis, inc in zip(output_ids, target_ac_tokens, batch_data['distance'], batch_data['includes']):
                    out, end_pos = filter_token_ids(out, start_id, end_id, unk_id)
                    tar, tar_end_pos = filter_token_ids(tar, start_id, end_id, unk_id)
                    if end_pos is None or tar_end_pos is None:
                        continue
                    obj = {'error_code_word_id': out, 'ac_code_word_id': tar, 'original_distance': dis, 'includes': inc}
                    records += [obj]

                steps += 1
                pbar.update(batch_size)

        # output_result, is_copy, value_output, pointer_output = model.forward_test(*parse_test_batch_data(batch_data))

    return records


def train_and_evaluate(batch_size, hidden_size, data_type, num_layers, dropout_p,
                       learning_rate, epoches, saved_name, load_name=None, epoch_ratio=1.0,
                       atte_position_type='position', clip_norm=80, logger_file_path=None,
                       do_sample_evaluate=False, print_output=False, addition_train=False, addition_train_remain_frac=1.0,
                       start_epoch=0, ac_copy_train=False, addition_epoch_ratio=0.4, ac_copy_radio=1.0,
                       do_multi_step_evaluate=False, max_sample_times=1, compile_file_path=None,
                       add_value_mask=False):
    valid_loss = 0
    test_loss = 0
    valid_accuracy = 0
    test_accuracy = 0
    valid_correct = 0
    test_correct = 0
    sample_valid_loss = 0
    sample_test_loss = 0
    sample_valid_accuracy = 0
    sample_test_accuracy = 0
    sample_valid_correct = 0
    sample_test_correct = 0

    save_path = os.path.join(config.save_model_root, saved_name)
    if load_name is not None:
        load_path = os.path.join(config.save_model_root, load_name)

    if logger_file_path is not None:
        init_a_file_logger(logger_file_path)

    begin_tokens = ['<BEGIN>']
    end_tokens = ['<END>']
    unk_token = '<UNK>'
    addition_tokens = ['<GAP>']
    vocabulary = create_common_error_vocabulary(begin_tokens=begin_tokens, end_tokens=end_tokens, unk_token=unk_token, addition_tokens=addition_tokens)
    special_token = pre_defined_c_tokens | pre_defined_c_library_tokens | set(vocabulary.begin_tokens) | \
                    set(vocabulary.end_tokens) | set(vocabulary.unk) | set(vocabulary.addition_tokens)
    vocabulary.special_token_ids = {vocabulary.word_to_id(t) for t in special_token}

    begin_tokens_id = [vocabulary.word_to_id(i) for i in begin_tokens]
    end_tokens_id = [vocabulary.word_to_id(i) for i in end_tokens]
    unk_token_id = vocabulary.word_to_id(unk_token)
    addition_tokens_id = [vocabulary.word_to_id(i) for i in addition_tokens]

    if data_type == 'deepfix':
        data_dict = load_deepfix_error_data()
        test_dataset = CCodeErrorDataSet(pd.DataFrame(data_dict), vocabulary, 'deepfix')
        info_output = "There are {} parsed data in the {} dataset".format(len(test_dataset), 'deepfix')
        print(info_output)
        # info(info_output)
        train_len = 100
        multi_step_no_target = True
    else:
        if is_debug:
            data_dict = load_common_error_data_sample_100()
        else:
            data_dict = load_common_error_data()
        datasets = [CCodeErrorDataSet(pd.DataFrame(dd), vocabulary, name) for dd, name in
                    zip(data_dict, ["train", "all_valid", "all_test"])]
        multi_step_no_target = False
        for d, n in zip(datasets, ["train", "val", "test"]):
            info_output = "There are {} parsed data in the {} dataset".format(len(d), n)
            print(info_output)
            # info(info_output)

        train_dataset, valid_dataset, test_dataset = datasets
        train_len = len(train_dataset)


    # combine_base_train_set = train_dataset
    if ac_copy_train:
        ac_copy_data_dict = create_copy_addition_data(data_dict[0]['ac_code_word_id'], data_dict[0]['includes'])
        ac_copy_dataset = CCodeErrorDataSet(pd.DataFrame(ac_copy_data_dict), vocabulary, 'ac_copy')
        info_output = "There are {} parsed data in the {} dataset".format(len(ac_copy_dataset), 'ac_copy')
        print(info_output)
        # combine_base_train_set = combine_base_train_set.combine_dataset(ac_copy_dataset)
    addition_dataset = None

    # model = SelfAttentionPointerNetworkModel(vocabulary_size=vocabulary.vocabulary_size, hidden_size=hidden_size,
    #                                  encoder_stack_num=encoder_stack_num, decoder_stack_num=decoder_stack_num,
    #                                  start_label=begin_tokens_id[0], end_label=end_tokens_id[0], dropout_p=dropout_p,
    #                                  num_heads=num_heads, normalize_type=normalize_type, MAX_LENGTH=MAX_LENGTH)
    model = RNNPointerNetworkModel(vocabulary_size=vocabulary.vocabulary_size, hidden_size=hidden_size,
                                     num_layers=num_layers, start_label=begin_tokens_id[0], end_label=end_tokens_id[0],
                                   dropout_p=dropout_p, MAX_LENGTH=MAX_LENGTH, atte_position_type=atte_position_type)
    print('model atte_position_type: {}'.format(atte_position_type))

    model = get_model(model)

    if load_name is not None:
        torch_util.load_model(model, load_path, map_location={'cuda:1': 'cuda:0'})

    loss_fn = create_combine_loss_fn(average_value=True)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = OpenAIAdam(model.parameters(), lr=learning_rate, schedule='warmup_linear', warmup=0.002, t_total=epoches * train_len//batch_size, max_grad_norm=clip_norm)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    if load_name is not None:
        # valid_loss, valid_accuracy, valid_correct = evaluate(model=model, dataset=valid_dataset, batch_size=batch_size,
        #                                       loss_function=loss_fn, vocab=vocabulary, add_value_mask=add_value_mask)
        # test_loss, test_accuracy, test_correct = evaluate(model=model, dataset=test_dataset, batch_size=batch_size,
        #                                     loss_function=loss_fn, vocab=vocabulary, add_value_mask=add_value_mask)
        if do_multi_step_evaluate:
            sample_valid_compile = 0
            first_sample_valid_compile = 0
            # sample_valid_compile, sample_valid_accuracy, sample_valid_correct, first_sample_valid_compile = evaluate_multi_step(model=model, dataset=valid_dataset,
            #          batch_size=batch_size, do_sample=True, vocab=vocabulary, print_output=print_output,
            #          max_sample_times=max_sample_times, file_path=compile_file_path)
            sample_test_compile, sample_test_accuracy, sample_test_correct, first_sample_test_compile = evaluate_multi_step(
                model=model, dataset=test_dataset, batch_size=batch_size, do_sample=True, vocab=vocabulary,
                print_output=print_output, max_sample_times=max_sample_times, file_path=compile_file_path,
                no_target=multi_step_no_target, add_value_mask=add_value_mask)
            evaluate_output = 'evaluate in multi step: sample valid compile: {}, sample test compile: {}, ' \
                              'first_sample_valid_compile: {}, first_sample_test_compile: {}' \
                              'sample valid accuracy: {}, sample test accuracy: {}, ' \
                              'sample valid correct: {}, sample test correct: {}'.format(
                sample_valid_compile, sample_test_compile, first_sample_valid_compile, first_sample_test_compile,
                sample_valid_accuracy, sample_test_accuracy, sample_valid_correct,
                sample_test_correct)
            print(evaluate_output)
            info(evaluate_output)

        if do_sample_evaluate:
            sample_valid_loss, sample_valid_accuracy, sample_valid_correct = evaluate(model=model, dataset=valid_dataset, batch_size=batch_size,
                                                  loss_function=loss_fn, do_sample=True, vocab=vocabulary,
                                                                                      print_output=print_output, add_value_mask=add_value_mask)
            sample_test_loss, sample_test_accuracy, sample_test_correct = evaluate(model=model, dataset=test_dataset, batch_size=batch_size,
                                                loss_function=loss_fn, do_sample=True, vocab=vocabulary,
                                                                                   print_output=print_output, add_value_mask=add_value_mask)
        evaluate_output = 'evaluate: valid loss of {}, test loss of {}, ' \
                          'valid_accuracy result of {}, test_accuracy result of {}, ' \
                          'valid correct result of {}, test correct result of {}, ' \
                          'sample valid loss: {}, sample test loss: {}, ' \
                          'sample valid accuracy: {}, sample test accuracy: {}, ' \
                          'sample valid correct: {}, sample test correct: {}'.format(
            valid_loss, test_loss, valid_accuracy, test_accuracy, valid_correct, test_correct,
            sample_valid_loss, sample_test_loss, sample_valid_accuracy, sample_test_accuracy, sample_valid_correct, sample_test_correct)
        print(evaluate_output)
        info(evaluate_output)
        pass

    for epoch in range(start_epoch, start_epoch+epoches):
        print('----------------------- in epoch {} --------------------'.format(epoch))
        combine_train_set = train_dataset
        if addition_train:
            addition_predict_dataset = train_dataset
            if addition_dataset is not None:
                addition_predict_radio = addition_epoch_ratio/(addition_epoch_ratio+1)
                addition_predict_dataset = addition_predict_dataset.combine_dataset(addition_dataset)
            else:
                addition_predict_radio = addition_epoch_ratio
            records = sample_better_output(model=model, dataset=addition_predict_dataset, batch_size=batch_size,
                                           vocab=vocabulary, do_sample=True, epoch_ratio=addition_predict_radio,
                                           add_value_mask=add_value_mask)

            addition_dict = create_addition_error_data(records)
            if addition_dataset is None:
                addition_dataset = CCodeErrorDataSet(pd.DataFrame(addition_dict), vocabulary, 'addition_train')
            else:
                addition_dataset.remain_samples(frac=addition_train_remain_frac)
                addition_dataset.add_samples(pd.DataFrame(addition_dict).sample(frac=1 - addition_train_remain_frac))
            info_output = "In epoch {}, there are {} parsed data in the {} dataset".format(epoch, len(addition_dataset), 'addition_train')
            print(info_output)
            combine_train_set = combine_train_set.combine_dataset(addition_dataset)

        if ac_copy_train:
            combine_train_set = combine_train_set.combine_dataset(ac_copy_dataset.remain_dataset(frac=ac_copy_radio))

        train_loss, train_accuracy = train(model=model, dataset=combine_train_set, batch_size=batch_size,
                                           loss_function=loss_fn, optimizer=optimizer, clip_norm=clip_norm,
                                           epoch_ratio=epoch_ratio, vocab=vocabulary, add_value_mask=add_value_mask)
        valid_loss, valid_accuracy, valid_correct = evaluate(model=model, dataset=valid_dataset, batch_size=batch_size,
                                              loss_function=loss_fn, vocab=vocabulary, add_value_mask=add_value_mask)
        test_loss, test_accuracy, test_correct = evaluate(model=model, dataset=test_dataset, batch_size=batch_size,
                                            loss_function=loss_fn, vocab=vocabulary, add_value_mask=add_value_mask)
        if do_sample_evaluate:
            sample_valid_loss, sample_valid_accuracy, sample_valid_correct = evaluate(model=model, dataset=valid_dataset, batch_size=batch_size,
                                                  loss_function=loss_fn, do_sample=True, vocab=vocabulary, add_value_mask=add_value_mask)
            sample_test_loss, sample_test_accuracy, sample_test_correct = evaluate(model=model, dataset=test_dataset, batch_size=batch_size,
                                                loss_function=loss_fn, do_sample=True, vocab=vocabulary, add_value_mask=add_value_mask)
        # scheduler.step(train_loss)
        # print('epoch {}: train loss: {}, accuracy: {}'.format(epoch, train_loss, train_accuracy))
        epoch_output = 'epoch {}: train loss of {}, valid loss of {}, test loss of {}, ' \
                       'train_accuracy result of {} valid_accuracy result of {}, test_accuracy result of {}' \
                       'valid correct result of {}, test correct result of {}' \
                       'sample valid loss: {}, sample test loss: {}, ' \
                       'sample valid accuracy: {}, sample test accuracy: {}' \
                       'sample valid correct: {}, sample test correct: {}'.format(
            epoch, train_loss, valid_loss, test_loss, train_accuracy, valid_accuracy, test_accuracy, valid_correct, test_correct,
            sample_valid_loss, sample_test_loss, sample_valid_accuracy, sample_test_accuracy, sample_valid_correct, sample_test_correct)
        print(epoch_output)
        info(epoch_output)

        if not is_debug:
            torch_util.save_model(model, save_path+str(epoch))


if __name__ == '__main__':
    # train_and_evaluate(batch_size=12, hidden_size=400, num_heads=3, encoder_stack_num=4, decoder_stack_num=4, num_layers=3, dropout_p=0,
    #                    learning_rate=6.25e-5, epoches=40, saved_name='SelfAttentionPointer.pkl', load_name='SelfAttentionPointer.pkl',
    #                    epoch_ratio=0.25, normalize_type=None, clip_norm=1, parallel=False, logger_file_path='log/SelfAttentionPointer.log')

    # train_and_evaluate(batch_size=16, hidden_size=400, num_heads=0, encoder_stack_num=0, decoder_stack_num=0, num_layers=3, dropout_p=0,
    #                    learning_rate=6.25e-5, epoches=40, saved_name='RNNPointerAllLoss.pkl', load_name='RNNPointerAllLoss.pkl',
    #                    epoch_ratio=0.25, normalize_type=None, atte_position_type='position', clip_norm=1, parallel=False,
    #                    do_sample_evaluate=True, print_output=True, logger_file_path='log/RNNPointerAllLossRecords.log')

    train_params = {'batch_size': 12, 'data_type': 'deepfix', 'learning_rate': 6.25e-5, 'epoches': 40,
                    'epoch_ratio': 0.25, 'start_epoch': 0, 'clip_norm': 1,
                    'do_sample_evaluate': False, 'print_output': True, 'addition_train': False,
                    'addition_train_remain_frac': 0.2, 'addition_epoch_ratio': 0.4,
                    'ac_copy_train': False, 'ac_copy_radio': 0.2,
                    'do_multi_step_evaluate': True, 'max_sample_times': 10, 'compile_file_path': 'log/tmp_compile.c'}


    # train_and_evaluate(batch_size=12, hidden_size=400, data_type='deepfix',
    #                    num_layers=3, dropout_p=0, learning_rate=6.25e-5, epoches=40,
    #                    saved_name='RNNPointerAllLossWithContentEmbeddingCombineTrainWeightCopyLoss.pkl',
    #                    load_name='RNNPointerAllLossWithContentEmbeddingCombineTrainWeightCopyLoss.pkl',
    #                    epoch_ratio=0.25, start_epoch=22, atte_position_type='content', clip_norm=1,
    #                    do_sample_evaluate=False, print_output=True,
    #                    logger_file_path='log/RNNPointerAllLossWithContentEmbeddingCombineTrainWeightCopyLossDeepfix.log',
    #                    addition_train=False, addition_train_remain_frac=0.2, addition_epoch_ratio=0.4,
    #                    ac_copy_train=False, ac_copy_radio=0.2, do_multi_step_evaluate=True, max_sample_times=10,
    #                    compile_file_path='log/tmp_compile.c', add_value_mask=False, multi_step_no_target=True)

    model_params1 = {'hidden_size': 400, 'num_layers': 3, 'dropout_p': 0, 'atte_position_type': 'content',
                     'add_value_mask': True,
                     }

    save_params1 = {'saved_name': 'RNNPointerAllLossWithContentEmbeddingCombineTrainWeightCopyLossWithTokenMask.pkl',
                    'load_name': 'RNNPointerAllLossWithContentEmbeddingCombineTrainWeightCopyLossWithTokenMask.pkl',
                    # 'load_name': None,
                    'logger_file_path': 'log/RNNPointerAllLossWithContentEmbeddingCombineTrainWeightCopyLossWithTokenMaskDeepfix.log',
                    }

    # train_and_evaluate(batch_size=12, hidden_size=400, data_type='deepfix', num_layers=3,
    #                    dropout_p=0, learning_rate=6.25e-5, epoches=40,
    #                    saved_name='RNNPointerAllLossWithContentEmbeddingCombineTrainWeightCopyLossWithTokenMask.pkl',
    #                    load_name='RNNPointerAllLossWithContentEmbeddingCombineTrainWeightCopyLossWithTokenMask.pkl',
    #                    epoch_ratio=0.25, start_epoch=22, atte_position_type='content', clip_norm=1,
    #                    do_sample_evaluate=False, print_output=True,
    #                    logger_file_path='log/RNNPointerAllLossWithContentEmbeddingCombineTrainWeightCopyLossWithTokenMaskDeepfix.log',
    #                    addition_train=False, addition_train_remain_frac=0.2, addition_epoch_ratio=0.4,
    #                    ac_copy_train=False, ac_copy_radio=0.2, do_multi_step_evaluate=True, max_sample_times=10,
    #                    compile_file_path='log/tmp_compile.c', add_value_mask=True)

    train_and_evaluate(**train_params, **model_params1, **save_params1)


