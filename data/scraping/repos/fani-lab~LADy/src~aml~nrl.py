import random, logging, pickle, pandas as pd, numpy as np, os, string

import nltk
import torch

from octis.models.model import *
from octis.dataset.dataset import Dataset
from octis.preprocessing.preprocessing import Preprocessing
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence

from .mdl import AbstractAspectModel
from cmn.review import Review

class Nrl(AbstractAspectModel):
    def __init__(self, octis_mdl, naspects, nwords, metrics):
        super().__init__(naspects, nwords)
        self.mdl = octis_mdl
        self.metrics = metrics

    def name(self): return 'octis.' + self.mdl.__class__.__name__.lower()
    def _seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True

    def load(self, path):
        self.mdl_out = load_model_output(f'{path}model.out.npz', vocabulary_path=None, top_words=self.nwords)
        self.mdl = pd.read_pickle(f'{path}model')
        self.dict = pd.read_pickle(f'{path}model.dict')
        self.cas = pd.read_pickle(f'{path}model.perf.cas')
        self.perplexity = pd.read_pickle(f'{path}model.perf.perplexity')

    def _create_ds(self, reviews_train, reviews_valid, output):
        if not os.path.isdir(output): os.makedirs(output)
        df_train = Review.to_df(reviews_train, w_augs=False)
        df_train['part'] = 'train'
        df_valid = Review.to_df(reviews_valid, w_augs=False)
        df_valid['part'] = 'val'
        df = pd.concat([df_train, df_valid])
        df.to_csv(f'{output}/corpus.tsv', sep='\t', encoding='utf-8', index=False, columns=['text', 'part'], header=None)

    def train(self, reviews_train, reviews_valid, settings, doctype, no_extremes, output):

        dataset = Dataset()
        try: dataset.load_custom_dataset_from_folder(f'{output}corpus')
        except:
            self._create_ds(reviews_train, reviews_valid, f'{output}corpus')
            dataset.load_custom_dataset_from_folder(f'{output}corpus')
        self.dict = dataset.get_vocabulary()
        self.mdl.hyperparameters.update(settings)
        self.mdl.hyperparameters.update({'num_topics': self.naspects})
        self.mdl.hyperparameters.update({'save_dir': None})#f'{output}model'})

        if 'bert_path' in self.mdl.hyperparameters.keys(): self.mdl.hyperparameters['bert_path'] = f'{output}corpus/'
        self.mdl.use_partitions = True
        self.mdl.update_with_test = True
        self.mdl_out = self.mdl.train_model(dataset, top_words=self.nwords)

        save_model_output(self.mdl_out, f'{output}model.out')
        # octis uses '20NewsGroup' as default corpus when no text passes! No warning?!
        self.cas = Coherence(texts=dataset.get_corpus(), topk=self.nwords, measure='u_mass', processes=settings['ncore']).score(self.mdl_out)
        pd.to_pickle(self.cas, f'{output}model.perf.cas')
        pd.to_pickle(self.dict, f'{output}model.dict')
        pd.to_pickle(self.mdl, f'{output}model')
        pd.to_pickle(self.perplexity, f'{output}model.perf.perplexity')

    def get_aspect_words(self, aspect_id, nwords):
        word_list = self.mdl_out['topics'][aspect_id]
        probs = []
        for w in word_list: probs.append(self.mdl_out['topic-word-matrix'][aspect_id][self.dict.index(w)])
        return list(zip(word_list, probs))

    def infer_batch(self, reviews_test, h_ratio, doctype, output):
        reviews_test_ = []
        reviews_aspects = []
        for r in reviews_test:
            r_aspects = [[w for a, o, s in sent for w in a] for sent in r.get_aos()]  # [['service', 'food'], ['service'], ...]
            if len(r_aspects[0]) == 0: continue  # ??
            if random.random() < h_ratio: r_ = r.hide_aspects()
            else: r_ = r
            reviews_aspects.append(r_aspects)
            reviews_test_.append(r_)

        #like in ctm (isinstance(self, CTM))
        if 'bert_model' in self.mdl.hyperparameters: _, test, input_size = self.mdl.preprocess(self.mdl.vocab, [], test=[r.get_txt() for r in reviews_test_], bert_model=self.mdl.hyperparameters['bert_model'])
        # like in neurallda isinstance(self, NeuralLDA)
        else: _, test, input_size = self.mdl.preprocess(self.mdl.vocab, [], test=[r.get_txt() for r in reviews_test_])
        test = self.mdl.inference(test)

        reviews_pred_aspects = [test['test-topic-document-matrix'][:, rdx] for rdx, _ in enumerate(reviews_test_)]
        pairs = []
        for i, r_pred_aspects in enumerate(reviews_pred_aspects):
            r_pred_aspects = [[(j, v) for j, v in enumerate(r_pred_aspects)]]
            pairs.extend(list(zip(reviews_aspects[i], self.merge_aspects_words(r_pred_aspects, self.nwords))))

        return pairs
