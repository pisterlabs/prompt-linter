import sys
from os import listdir
from os.path import isfile, join
import pickle
import hashlib
import time

import numpy as np
import torch

PATH_BERT = '../../pytorch-pretrained-BERT'
sys.path.insert(0, PATH_BERT)
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel
from .encoder import BaseEncoder


class GPTEncoder(BaseEncoder):

    def __init__(self, model_name='openai-gpt', encode_capacity=3000, path_cache='./cache'):
        super(GPTEncoder, self).__init__(model_name, encode_capacity, path_cache)

    def construct_encoder(self):
        model = OpenAIGPTModel.from_pretrained(self.model_name)
        model.cuda()
        model = torch.nn.DataParallel(model)
        model.eval()
        tokenizer = OpenAIGPTTokenizer.from_pretrained(self.model_name)
        print("Model and tokenzier are constructed!")
        return model, tokenizer

    def convert_sentences_to_features(self, sentences, seq_length):
        """Convert sentence into Tensor"""

        num_sent = len(sentences)
        input_ids = np.zeros((num_sent, seq_length), dtype=np.int32)
        seq_lens = []

        for idx, sent in enumerate(sentences):
            tokens = self.tokenizer.tokenize(sent)
            seq_lens.append(len(tokens)) # record position of the last token
            tokens = tokens[0:min(seq_length, len(tokens))]  # truncate tokens longer than seq_length
            input_ids[idx, :len(tokens)] = np.array(self.tokenizer.convert_tokens_to_ids(tokens), dtype=np.int32)
            assert len(input_ids[idx]) == seq_length

        return input_ids, np.array(seq_lens)

    def encode(self, sentences, heads, head_size, location):
        ts = time.time()
        # self.model.eval()

        if not self.flag_cache_save:
            output = []
            for i, sent in enumerate(sentences):
                hask_key = hashlib.sha256(sent.encode()).hexdigest()
                output.append(self.cache[hask_key])
            output = np.array(output)
        else:
            print('encoding with model', len(sentences))
            mini_batch_size = self.get_mini_batch_size(sentences)
            idx = 0
            list_output = []
            while idx < len(sentences):
                mini_batch = sentences[idx:min(idx + mini_batch_size, len(sentences))]
                seq_length = max([len(tokens) for tokens in mini_batch])

                # ====== Convert to Tensor ====== #
                indexed_tokens, seq_lens = self.convert_sentences_to_features(mini_batch, seq_length)
                tokens_tensor = torch.Tensor([indexed_tokens]).long()
                tokens_tensor = tokens_tensor.cuda()

                # ====== Encode Tokens ====== #
                encoded_layers, self_attention_layers = self.model(tokens_tensor)
                torch.cuda.synchronize()

                n_sent = encoded_layers[0].shape[0]
                hid_dim = encoded_layers[0].shape[2]

                rows = np.tile(np.arange(n_sent).T, (hid_dim, 1)).T[:, np.newaxis, :]
                tokens = np.tile(seq_lens, (hid_dim, 1)).T[:, np.newaxis, :]
                columns = np.tile(np.arange(hid_dim), (n_sent, 1, 1))

                if location == 'fc':
                    output = np.array([np.squeeze(layer[rows, tokens, columns]).detach().cpu().numpy() for layer in encoded_layers])
                elif location == 'head':
                    output = np.array([np.squeeze(layer[rows, tokens, columns]).detach().cpu().numpy() for layer in self_attention_layers])

                if len(output.shape) == 2:
                    output = output.reshape(output.shape[0], -1, output.shape[1])

                output = np.swapaxes(output, 0, 1)
                list_output.append(output)

                # ====== Construct Cache ====== #
                temp_cache = {}
                for i, sent in enumerate(mini_batch):
                    hask_key = hashlib.sha256(sent.encode()).hexdigest()
                    temp_cache[hask_key] = output[i]
                self.cache.update(temp_cache)

                idx += mini_batch_size
                self.count += mini_batch_size
            output = np.concatenate(list_output, 0)

        te = time.time()
        embedding = self.get_multi_head_embedding(output, heads, head_size)
        return embedding



if __name__ == '__main__':
    model = OpenAIGPTModel('bert-base-uncased')
    model.prepare('Length')
    model.construct_encoder()


