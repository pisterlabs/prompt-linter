# -*- coding: utf-8 -*-
"""
Assess model performance
"""
from __future__ import print_function, division

import os

from nltk import sent_tokenize
from utils import replace_sents, pk_load
from evaluator import Evaluator
from coherence_probability import ProbabilityVector


class Assessment(object):
    def __init__(self, corpus, pv, ev):
        self.corpus = self._preprocess(corpus) + self._label_corpus(corpus)
        assert type(pv) == ProbabilityVector
        assert type(ev) == Evaluator
        self.pv = pv
        self.ev = ev

    def _preprocess(self, corpus):
        res = []
        for text in corpus:
            text = '. '.join(text.split('.'))
            res.append((text, 1))
        return res

    def _label_corpus(self, corpus):
        res = []
        for text in corpus:
            text = '. '.join(text.split('.'))
            remove_one = replace_sents(text, 1)[0]
            res.append((remove_one, -1))
        return res

    def assess_pv(self, text):
        if len(sent_tokenize(text)) <= 1:
            return -1
        pb = self.pv.evaluate_coherence(text)[0]
        if pb < self.pv.mean:
            return -1
        elif self.pv.mean <= pb <= self.pv.mean + 2 * self.pv.std:
            return 1
        else:
            return 1

    def assess_ev(self, text):
        rank = self.ev.evaluate_coherence(text)[0]
        if rank < 0.2:
            return -1
        elif 0.2 <= rank < 1:
            return 1
        else:
            return 1

    def assess_all(self):
        ev_right, pv_right, length = 0, 0, len(self.corpus)
        cnt = 0
        for text, label in self.corpus:
            ev_res, pv_res = None, None
            cnt += 1
            try:
                ev_res = self.assess_ev(text)
                pv_res = self.assess_pv(text)
            except Exception:
                print(text)
            else:
                print('{}/{}'.format(cnt, length))
            if ev_res == label:
                ev_right += 1
            if pv_res == label:
                pv_right += 1
        return ev_right / length, pv_right / length


if __name__ == '__main__':
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    pv = pk_load(os.path.join(cur_dir, 'pickles', 'pv.pkl'))
    ev = pk_load(os.path.join(cur_dir, 'pickles', 'ev.pkl'))
    with open(os.path.join(cur_dir, 'corpus', 'test.txt')) as f:
        testtxt = f.read().split('////')
        assess = Assessment(testtxt[:2], pv, ev)
        print(assess.assess_all())
