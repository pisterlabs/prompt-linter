''' Utils for io, language, connectivity graphs etc '''

import os
import sys
import re
import string
import json
import time
import math
from collections import Counter
import numpy as np
import networkx as nx


# padding, unknown word, end of sentence
base_vocab = ['<PAD>', '<UNK>', '<EOS>']
padding_idx = base_vocab.index('<PAD>')

def load_nav_graphs(scans):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    graphs = {}
    for scan in scans:
        with open('connectivity/%s_connectivity.json' % scan) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i,item in enumerate(data):
                if item['included']:
                    for j,conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                    item['pose'][7], item['pose'][11]])
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs


def dump_transformer_index(encoder_type, splits):
    if encoder_type == 'bert' or encoder_type == 'vlbert':
        dump_bert_index(splits)
    elif encoder_type == 'gpt':
        dump_gpt_index(splits)
    else:
        raise NotImplementedError

def dump_gpt_index(splits):
    from pytorch_pretrained_bert import OpenAIGPTTokenizer
    tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    #splits = ['train', 'val_seen', 'val_unseen', 'test']

    for split in splits:
        data = load_datasets([split], encoder_type='lstm') # here we use lstm dataset to preprocess the data,
        indexed_tokens = []
        for item in data:
            for instr in item['instructions']:
                tokenized_text = tokenizer.tokenize(instr)
                tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
                indexed_tokens.append('_'.join([str(i) for i in tokens]))
        write_vocab(indexed_tokens, 'tasks/R2R/data/R2R_%s_gpt.txt' % split)


def dump_bert_index(splits):
    from pytorch_pretrained_bert import BertTokenizer
    from nltk.tokenize import sent_tokenize

    tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
    #splits = ['train', 'val_seen', 'val_unseen', 'test']

    for split in splits:
        data = load_datasets([split] ,encoder_type='lstm') # here we use lstm dataset to preprocess the data,
        indexed_tokens = []
        for item in data:
            for instr in item['instructions']:
                sents = sent_tokenize(instr)
                instr = '[CLS] ' + (' [SEP] '.join(sents)) + ' [SEP]'
                tokenized_text = tokenizer.tokenize(instr)
                tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
                indexed_tokens.append('_'.join([str(i) for i in tokens]))
        write_vocab(indexed_tokens, 'tasks/R2R/data/R2R_%s_bert.txt' % split)


def load_datasets(splits, encoder_type):
    data = []
    for split in splits:
        with open('tasks/R2R/data/R2R_%s.json' % split) as f:
            data += json.load(f)

        if encoder_type in ['bert', 'gpt','vlbert']:
            #filename = 'tasks/R2R/data/R2R_%s_%s.txt' % (split, encoder_type)
            if encoder_type == 'bert' or encoder_type == 'vlbert':
                filename = 'tasks/R2R/data/R2R_%s_bert.txt' % (split)
            else:
                filename = 'tasks/R2R/data/R2R_%s_%s.txt' % (split, encoder_type)
            if not os.path.exists(filename):
                dump_transformer_index(encoder_type, [split])
            transformer_index = read_vocab(filename)
            j=0
            err_items = []
            for k, item in enumerate(data):
                for i, instr in enumerate(item['instructions']):
                    item['instructions'][i] = transformer_index[j]
                    if not transformer_index[j]:
                        err_items.append(k)
                    j+=1
            assert j == len(transformer_index)
            for k in err_items[::-1]:
                data.pop(k)
    return data


class SplitTokenizer():
    def __init__(self, pad_idx=0, encoding_length=20):
        self.encoding_length = encoding_length
        self.pad_idx=pad_idx

    def encode_sentence(self, sentence):
        #print(sentence)
        encoding = [] if len(sentence.strip())==0 else [int(i) for i in sentence.strip().split('_')]
        if len(encoding) < self.encoding_length:
            encoding += [self.pad_idx] * (self.encoding_length-len(encoding))
        return np.array(encoding[:self.encoding_length])


class Tokenizer(object):
    ''' Class to tokenize and encode a sentence. '''
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)') # Split on any non-alphanumeric character

    def __init__(self, vocab=None, encoding_length=20):
        self.encoding_length = encoding_length
        self.vocab = vocab
        self.word_to_index = {}
        if vocab:
            for i,word in enumerate(vocab):
                self.word_to_index[word] = i

    def split_sentence(self, sentence):
        ''' Break sentence into a list of words and punctuation '''
        toks = []
        for word in [s.strip().lower() for s in self.SENTENCE_SPLIT_REGEX.split(sentence.strip()) if len(s.strip()) > 0]:
            # Break up any words containing punctuation only, e.g. '!?', unless it is multiple full stops e.g. '..'
            if all(c in string.punctuation for c in word) and not all(c in '.' for c in word):
                toks += list(word)
            else:
                toks.append(word)
        return toks

    def encode_sentence(self, sentence):
        if len(self.word_to_index) == 0:
            sys.exit('Tokenizer has no vocab')

        encoding = []
        for word in self.split_sentence(sentence)[::-1]: # reverse input sentences
            if word in self.word_to_index:
                encoding.append(self.word_to_index[word])
            else:
                encoding.append(self.word_to_index['<UNK>'])
        encoding.append(self.word_to_index['<EOS>'])

        if len(encoding) < self.encoding_length:
            encoding += [self.word_to_index['<PAD>']] * (self.encoding_length-len(encoding))

        return np.array(encoding[:self.encoding_length])

    def decode_sentence(self, encoding):
        sentence = []
        for ix in encoding:
            if ix == self.word_to_index['<PAD>']:
                break
            else:
                sentence.append(self.vocab[ix])
        return " ".join(sentence[::-1]) # unreverse before output


def build_vocab(splits=['train'], min_count=5, start_vocab=base_vocab):
    ''' Build a vocab, starting with base vocab containing a few useful tokens. '''
    count = Counter()
    t = Tokenizer()
    data = load_datasets(splits, encoder_type='lstm')#, False)
    for item in data:
        for instr in item['instructions']:
            count.update(t.split_sentence(instr))

    vocab = list(start_vocab)
    for word,num in count.most_common():
        if num >= min_count:
            vocab.append(word)
        else:
            break
    return vocab


def write_vocab(vocab, path):
    print('Writing vocab of size %d to %s' % (len(vocab),path))
    with open(path, 'w') as f:
        for word in vocab:
            f.write("%s\n" % word)


def read_vocab(path):
    with open(path) as f:
        vocab = [word.strip() for word in f.readlines()]
    return vocab


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def to_contiguous(tensor):  # jolin
    if tensor.is_contiguous(): return tensor
    else: return tensor.contiguous()


def clip_gradient(optimizer, grad_clip=0.1):  # jolin
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)



def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def dump_get_navigable():
    from pytorch_pretrained_bert import BertTokenizer
    from nltk.tokenize import sent_tokenize

    tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
    splits = ['train', 'val_seen', 'val_unseen', 'test']
    for split in splits:
        data = load_datasets([split] ,False)
        indexed_tokens = []
        for item in data:
            for instr in item['instructions']:
                sents = sent_tokenize(instr)
                instr = '[CLS] ' + (' [SEP] '.join(sents)) + ' [SEP]'
                tokenized_text = tokenizer.tokenize(instr)
                tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
                indexed_tokens.append('_'.join([str(i) for i in tokens]))
        write_vocab(indexed_tokens, 'tasks/R2R/data/R2R_%s_bert.txt' % split)


def _loc_distance(loc):
    return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)


def preprocess_get_pano_states(navigable_locs_path = "tasks/R2R/data/navigable_locs.json"):
    if os.path.exists(navigable_locs_path):
        return
    image_w = 640
    image_h = 480
    vfov = 60
    import sys
    sys.path.append('build')
    import MatterSim
    from collections import defaultdict

    sim = MatterSim.Simulator()
    sim.setRenderingEnabled(False)
    sim.setDiscretizedViewingAngles(True)
    sim.setCameraResolution(image_w, image_h)
    sim.setCameraVFOV(math.radians(vfov))
    sim.init()

    splits = ['train', 'val_seen', 'val_unseen', 'test']
    graphs = {}
    for split in splits:
        data = load_datasets([split], encoder_type='lstm')
        for item in data:
            # print(item.keys())
            # print("")
            scan = item["scan"]
            if scan in graphs:
                continue
            graphs[scan] = {}
            with open('connectivity/%s_connectivity.json' % scan) as f:
                data = json.load(f)
                for i, item in enumerate(data):
                    if item['included']:
                        viewpointId = item['image_id']
                        sim.newEpisode(scan, viewpointId, 0, 0)
                        state = sim.getState()

                        initViewIndex = state.viewIndex
                        # 1. first look down, turning to relViewIndex 0
                        elevation_delta = -(state.viewIndex // 12)
                        for _ in range(int(abs(elevation_delta))):
                            ''' Make possibly more than one elevation turns '''
                            sim.makeAction(0, 0, np.sign(elevation_delta))

                        adj_dict = {}
                        for relViewIndex in range(36):
                            state = sim.getState()
                            absViewIndex = state.viewIndex
                            for loc in state.navigableLocations[1:]:
                                distance = _loc_distance(loc)
                                if (loc.viewpointId not in adj_dict or
                                    distance < adj_dict[loc.viewpointId]['distance']):
                                    adj_dict[loc.viewpointId] = {
                                        'absViewIndex': absViewIndex,
                                        'nextViewpointId': loc.viewpointId,
                                        'loc_rel_heading': loc.rel_heading,
                                        'loc_rel_elevation': loc.rel_elevation,
                                        'distance': distance}
                            if (relViewIndex + 1) % 12 == 0:
                                sim.makeAction(0, 1, 1)  # Turn right and look up
                            else:
                                sim.makeAction(0, 1, 0)  # Turn right
                        # 3. turn back to the original view
                        for _ in range(int(abs(- 2 - elevation_delta))):
                            ''' Make possibly more than one elevation turns '''
                            sim.makeAction(0, 0, np.sign(- 2 - elevation_delta))

                        state = sim.getState()
                        assert state.viewIndex == initViewIndex

                        absViewIndex2points = defaultdict(list)
                        for vpId, point in adj_dict.items():
                            absViewIndex2points[point['absViewIndex']].append(vpId)
                        graphs[scan][viewpointId]=(adj_dict, absViewIndex2points)
        print('prepare cache for', split, 'done')
    with open(navigable_locs_path, 'w') as f:
        json.dump(graphs, f)


def current_best(df, v_id, best_score_name):
    if best_score_name == 'sr_sum':
        return  df['val_seen success_rate'][v_id] + df['val_unseen success_rate'][v_id]
    elif best_score_name == 'spl_sum':
        return  df['val_seen spl'][v_id] + df['val_unseen spl'][v_id]
    elif best_score_name == 'spl_unseen':
        return  df['val_unseen spl'][v_id]
    elif best_score_name == 'sr_unseen':
        return  df['val_unseen success_rate'][v_id]


def show_path_steps_len(splits):
    ''' histogram of path length in the whole dataset '''
    import matplotlib.pyplot as plt
    path_lens = []
    for split in splits:
        data = load_datasets([split], False)
        path_lens.extend([len(item['path']) for item in data])
        print(len(data))
    print('min steps', min(path_lens),'max steps', max(path_lens))
    plt.hist(path_lens,
             bins=[i for i in range(min(path_lens), max(path_lens) + 1)])  # arguments are passed to np.histogram
    plt.title("Histogram with '%d-%d' bins" % ((min(path_lens), max(path_lens))))
    plt.show()


def show_max_navigable():
    navigable_locs_path = "tasks/R2R/data/navigable_locs.json"
    with open(navigable_locs_path, 'r') as f:
        nav_graphs = json.load(f)

    max_navigable = 0
    for scan in nav_graphs:
        for viewpointId in nav_graphs[scan]:
            adj_dict, absViewIndex2points = nav_graphs[scan][viewpointId]
            if max_navigable < len(adj_dict):
                max_navigable = len(adj_dict)
    print(max_navigable)


def generate_multisent_to_dataset():
    from nltk.tokenize import sent_tokenize
    import copy
    splits = ['train', 'val_seen', 'val_unseen', 'test']

    counter = ([],[])
    for split in splits:
        new_data = []
        data = load_datasets([split] ,encoder_type='lstm') # here we use lstm dataset to preprocess the data,
        for item in data:
            for i,instr in enumerate(item['instructions']):
                new_item = copy.deepcopy(item)
                sents = sent_tokenize(instr)
                new_item['path_id'] = "%s_%d"%(item['path_id'],i)
                new_item['instructions'] = sents
                new_data.append(new_item)
                counter[0].append(len(sents))
                counter[1].append(max([len(sent) for sent in sents]))
        with open("tasks/R2R/data/R2R_%s_multisent.json"%split, 'w') as fout:
            json.dump(new_data, fout, indent=2, separators=[',',':'])
    print(max(counter[0]), max(counter[1]))

if __name__ == '__main__':
    # show_path_steps_len(['train_subgoal', 'val_seen_subgoal', 'val_unseen_subgoal'])
    # show_path_steps_len(['train', 'val_seen', 'val_unseen'])
    show_max_navigable()
