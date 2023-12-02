from __future__ import annotations

import attr
import ast
import numpy as np
import pandas as pd
from typing import Any, Optional, Union
from pydantic import BaseModel, Extra
from copy import deepcopy
from rank_bm25 import BM25Okapi
from datasets import Dataset
from pathlib import Path

from langchain.prompts.example_selector.base import BaseExampleSelector
from prompts.base import ExampleTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings
from selector.base import CommonSelectorArgs, SelectorUtilsMixin
from selector.greedy import decomposed_coverage_greedy
from tools.track import track

@attr.s(auto_attribs=True)
class LFCoverageSelectorArgs(CommonSelectorArgs):
    data_root: Path
    dataset: str
    split: str
    seed: int

    def get_name(self):
        # return f'split_{self.split}-seed{self.seed}'
        return f'split_{self.split}'

Args = LFCoverageSelectorArgs

class LFCoverageSelector(BaseExampleSelector, BaseModel):
    args: Args
    example_template: ExampleTemplate
    demo_candidates: Dataset
    queryqid2idx: dict[str, int] = None
    shot_idxs_l: Optional[Union[np.ndarray, list[np.ndarray]]] = None

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True

    def add_example(self, example: dict[str, str]) -> Any:
        ...

    @classmethod
    def get_shot_idxes(
        cls, args: Args, query_ex, shots_df, candqid2idx, cand_lens=None, max_len=-1
    ):
        query_qid = query_ex['qid'].replace('geoquery', 'geo880')
        row = shots_df.query(f'input_qid == "{query_qid}"').iloc[0]
        prompt_qids_raw = [
            qid.replace('geo880', 'geoquery')
            for qid in ast.literal_eval(row.prompt_qids_raw)]
        cand2pos = {
            qid.replace('geo880', 'geoquery'): i
            for i, qid in enumerate(ast.literal_eval(row.prompt_qids_sorted))}
        shot_qids = []
        rem_len = max_len if max_len > 0 else 1000000
        for next_qid in prompt_qids_raw:
            if next_qid not in candqid2idx:
                continue
            if cand_lens and cand_lens[candqid2idx[next_qid]] > rem_len:
                break
            elif cand_lens:
                rem_len -= cand_lens[candqid2idx[next_qid]]
            shot_qids.append(next_qid)
            if len(shot_qids) >= min(args.n_shots, len(prompt_qids_raw)):
                break
        shot_qids = sorted(shot_qids, key=lambda qid: cand2pos[qid])
        shot_idxs = np.array([candqid2idx[qid]
                              for qid in shot_qids])
        return shot_idxs

    def select_examples(self, input_variables: dict[str, str]) -> list[dict]:
        shot_idxs = self.shot_idxs_l[self.queryqid2idx[input_variables['qid']]]
        return self.demo_candidates.select(shot_idxs)

    @classmethod
    def from_examples(
        cls,
        args: Args,
        examples: list[dict],
        example_template: ExampleTemplate,
        query_examples: list[dict] = None,
        enc_len_fn: Any = None,
        max_len: int = -1,
        subtract_gen_len: bool = False,
        progress_bar: bool = False,
    ) -> LFCoverageSelector:
        query_examples = query_examples or []
        candqid2idx = {ex['qid']: i for i, ex in enumerate(examples)}
        queryqid2idx = {ex['qid']: i for i, ex in enumerate(query_examples)}

        max_len = max_len if max_len > 0 else 1000000
        cand_lens = [enc_len_fn(example_template.format(**ex)) if enc_len_fn else 0
                    for ex in examples]
        query_lens = [enc_len_fn(example_template.format(**ex, test=True)) if enc_len_fn else 0
                    for ex in query_examples]
        completed_query_lens = [enc_len_fn(example_template.format(**ex)) if enc_len_fn else 0
                    for ex in query_examples]

        split, dataset = args.split, args.dataset
        shots_df = pd.read_csv(args.data_root / f'{dataset}/{split}/random_seed_{args.seed}/prompt_qids.csv', index_col=0)

        print('Finding shots...')
        n_queries = len(query_examples)
        query_iter = track(range(n_queries), description='Finding shots', total=n_queries) if progress_bar else range(n_queries)
        shot_idxs_l = []
        for idx in query_iter:
            if not subtract_gen_len:
                _max_len = max_len - query_lens[idx]
            else:
                _max_len = max_len - completed_query_lens[idx] - 4
            shot_idxs = cls.get_shot_idxes(
                args, query_examples[idx], shots_df, candqid2idx, cand_lens, _max_len)
            shot_idxs_l.append(shot_idxs)
        print(f'Average number of shots: {np.mean([len(shot_idxs) for shot_idxs in shot_idxs_l])}')
        return cls(
            args=args,
            example_template=example_template,
            demo_candidates=examples,
            queryqid2idx=queryqid2idx,
            shot_idxs_l=shot_idxs_l,
        )