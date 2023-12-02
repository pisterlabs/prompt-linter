import numpy as np, pandas as pd, random, os
from typing import List

import torch
from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessingStopwords
from contextualized_topic_models.evaluation.measures import CoherenceUMASS

from cmn.review import Review
from .mdl import AbstractAspectModel, BatchPairsType

class Ctm(AbstractAspectModel):
    def __init__(self, naspects, nwords, contextual_size, nsamples):
        super().__init__(naspects, nwords)
        self.contextual_size = contextual_size
        self.nsamples = nsamples

    def _seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True

    def load(self, path):
        from natsort import natsorted
        self.tp = pd.read_pickle(f'{path}model.tp')
        self.mdl = CombinedTM(bow_size=len(self.tp.vocab), contextual_size=self.contextual_size, n_components=self.naspects)
        files = list(os.walk(f'{path}model'))

        print(f"{files[-1][0]}/{natsorted(files[-1][-1])[-1]}")
        self.mdl.load(files[-1][0], epoch=int(natsorted(files[-1][-1])[-1].replace('epoch_', '').replace('.pth', '')))
        # self.mdl.load(files[-1][0], epoch=settings['num_epochs'] - 1) # based on validation set, we may have early stopping, so the final model may be saved for earlier epoch
        self.dict = pd.read_pickle(f'{path}model.dict')
        self.cas = pd.read_pickle(f'{path}model.perf.cas')
        self.perplexity = pd.read_pickle(f'{path}model.perf.perplexity')

    def train(self, reviews_train, reviews_valid, settings, doctype, no_extremes, output):
        corpus_train, self.dict = super(Ctm, self).preprocess(doctype, reviews_train, no_extremes)
        corpus_train = [' '.join(doc) for doc in corpus_train]

        self._seed(settings['seed'])
        self.tp = TopicModelDataPreparation(settings['bert_model'])

        processed, unprocessed, vocab, _ = WhiteSpacePreprocessingStopwords(corpus_train, stopwords_list=[]).preprocess()#we already preproccess corpus in super()
        training_dataset = self.tp.fit(text_for_contextual=unprocessed, text_for_bow=processed)
        self.dict = self.tp.vocab

        valid_dataset = None
        # bug when we have validation=> RuntimeError: mat1 and mat2 shapes cannot be multiplied (5x104 and 94x100)
        # File "C:\ProgramData\Anaconda3\envs\lady\lib\site-packages\contextualized_topic_models\models\ctm.py", line 457, in _validation
        if len(reviews_valid) > 0:
            corpus_valid, _ = super(Ctm, self).preprocess(doctype, reviews_valid, no_extremes)
            corpus_valid = [' '.join(doc) for doc in corpus_valid]
            processed_valid, unprocessed_valid, _, _ = WhiteSpacePreprocessingStopwords(corpus_valid, stopwords_list=[]).preprocess()
            valid_dataset = self.tp.transform(text_for_contextual=unprocessed_valid, text_for_bow=processed_valid)

        self.mdl = CombinedTM(bow_size=len(self.tp.vocab),
                              contextual_size=settings['contextual_size'],
                              n_components=self.naspects,
                              num_epochs=settings['num_epochs'],
                              num_data_loader_workers=settings['ncore'],
                              batch_size=min([settings['batch_size'], len(training_dataset), len(valid_dataset) if valid_dataset else np.inf]))
                            # drop_last=True!! So, for small train/valid sets, it raises devision by zero in val_loss /= samples_processed

        self.mdl.fit(train_dataset=training_dataset, validation_dataset=valid_dataset, verbose=True, save_dir=f'{output}model', )
        self.cas = CoherenceUMASS(texts=[doc.split() for doc in processed], topics=self.mdl.get_topic_lists(self.nwords)).score(topk=self.nwords, per_topic=True)

        # self.mdl.get_doc_topic_distribution(training_dataset, n_samples=20)
        # log_perplexity = -1 * np.mean(np.log(np.sum(bert, axis=0)))
        # self.perplexity = np.exp(log_perplexity)

        pd.to_pickle(self.dict, f'{output}model.dict')
        pd.to_pickle(self.tp, f'{output}model.tp')
        pd.to_pickle(self.cas, f'{output}model.perf.cas')
        pd.to_pickle(self.perplexity, f'{output}model.perf.perplexity')
        self.mdl.save(f'{output}model')

    def get_aspect_words(self, aspect_id, nwords): return self.mdl.get_word_distribution_by_topic_id(aspect_id)[:nwords]

    def infer_batch(self, reviews_test: List[Review], h_ratio, doctype, output):
        reviews_test_ = []
        reviews_aspects: List[List[List[int]]] = []
        for r in reviews_test:
            r_aspects = [[w for a, o, s in sent for w in a] for sent in r.get_aos()]  # [['service', 'food'], ['service'], ...]

            if len(r_aspects[0]) == 0: continue  # ??
            if random.random() < h_ratio: r_ = r.hide_aspects()
            else: r_ = r

            reviews_aspects.append(r_aspects)
            reviews_test_.append(r_)

        corpus_test, _ = super(Ctm, self).preprocess(doctype, reviews_test_)
        corpus_test = [' '.join(doc) for doc in corpus_test]

        processed, unprocessed, vocab, _ = WhiteSpacePreprocessingStopwords(corpus_test, stopwords_list=[]).preprocess()
        testing_dataset = self.tp.transform(text_for_contextual=unprocessed, text_for_bow=processed)
        reviews_pred_aspects = self.mdl.get_doc_topic_distribution(testing_dataset, n_samples=self.nsamples)
        pairs: BatchPairsType = []
        for i, r_pred_aspects in enumerate(reviews_pred_aspects):
            r_pred_aspects = [[(j, v) for j, v in enumerate(r_pred_aspects)]]
            pairs.extend(list(zip(reviews_aspects[i], self.merge_aspects_words(r_pred_aspects, self.nwords))))

        return pairs
