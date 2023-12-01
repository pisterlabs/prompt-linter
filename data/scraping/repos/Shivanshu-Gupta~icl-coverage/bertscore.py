from __future__ import annotations

import attr
import torch
import numpy as np
from typing import Any
from collections import defaultdict
from pydantic import BaseModel, Extra
from more_itertools import chunked
from datasets import Dataset
from bert_score.utils import get_tokenizer, get_model, model2layers

from langchain.prompts.example_selector.base import BaseExampleSelector
from prompts.base import ExampleTemplate
from selector.base import CommonSelectorArgs, SelectorUtilsMixin
from selector.greedy import decomposed_coverage_greedy
from tools.track import track

def embed(sents, model, tokenizer, idf_dict, device, contexts=None):
    print(f'Embedding {len(sents)} sentences ...')
    from bert_score.utils import get_bert_embedding, sent_encode
    batch_size = 64
    embs_l, idfs_l = [], []
    for batch_start in track(range(0, len(sents), batch_size)):
        sen_batch = sents[batch_start:batch_start+batch_size]
        embs, masks, padded_idf = get_bert_embedding(
            sen_batch, model, tokenizer, idf_dict, device=device, all_layers=False
        )
        embs = embs.cpu().numpy()
        masks = masks.cpu().numpy()
        padded_idf = padded_idf.cpu().numpy()
        for i in range(len(sen_batch)):
            sequence_len = masks[i].sum().item()
            emb = embs[i, :sequence_len]
            idf = padded_idf[i, :sequence_len]
            if contexts is not None:
                ctx = sent_encode(tokenizer, contexts[1])
                emb = emb[len(ctx):]
                idf = idf[len(ctx):]
            emb /= np.linalg.norm(emb, axis=1, keepdims=True)
            embs_l.append(emb)
            idfs_l.append(idf)
    return embs_l, idfs_l

def cat_tensors(tensors, dim=0):
    return torch.cat([t for t in tensors], dim=dim)

def list_tensors(tensors, sizes, dim=0):
    return torch.split(tensors, sizes, dim=dim)

    # return [t[s:e] for t, s, e in zip(tensors, starts, ends)]

def compute_sims_packed(query_embs, query_mask, cand_embs, cand_mask):
    cand_embs = cat_tensors(cand_embs)  # [kd]*c -> Kd
    cand_mask = cat_tensors(cand_mask)  # [k]*c -> K
    if query_embs.ndim == 3:
        # sims = torch.einsum('bld,bl,ckd,ck->bclk', query_embs, query_mask, cand_embs, cand_mask)
        sims = torch.einsum('bld,Kd->bKl', query_embs, cand_embs)
        mask = torch.einsum('bl,ck->bKl', query_mask, cand_mask)
        sims[mask == 0] = sims.min() - 1e5
    else:
        if query_mask is None:
            query_mask = torch.ones(query_embs.shape[0], device=query_embs.device)
        # sims = torch.einsum('ld,l,ckd,ck->clk', query_embs, query_mask, cand_embs, cand_mask)
        sims = torch.einsum('ld,Kd->Kl', query_embs, cand_embs)
        mask = torch.einsum('l,K->Kl', query_mask, cand_mask)
        sims[mask == 0] = sims.min() - 1e5
    return sims

def list_sims_to_token_recalls(sims, query_idf):
    # sims: [blk]*c or [lk]*c
    # query_idf: bl
    recall = [s.max(axis=-1).values for s in sims]   # [blk]*c -> [bl]*c or [lk]*c -> [l]*c
    recall = torch.stack(recall[..., None, :], dim=-2)  # [bl]*c -> bcl or [l]*c -> cl
    recall_scale = query_idf / query_idf.sum(axis=-1, keepdims=True)  # bl / b1 -> bl or l / 1 -> l
    recall_scale = recall_scale.unsqueeze(-2)    # b1l or 1l
    recall = recall * recall_scale # bcl * b1l -> bcl or cl * 1l -> cl
    return recall

def list_sims_to_token_precision(sims, cand_idf):
    # sims: [blk]*c or [lk]*c
    # query_idf: bl

    precision = sims.max(axis=-2).values    # bclk -> bck or clk -> ck
    precision_scale = cand_idf / cand_idf.sum(axis=-1, keepdims=True)    # ck / c1 -> ck or k / 1 -> k
    precision = precision * precision_scale # bck * ck -> bck or ck * k -> ck
    return precision

def compute_sims(query_embs, query_mask, cand_embs, cand_mask):
    if query_embs.ndim == 3:
        # sims = torch.einsum('bld,bl,ckd,ck->bclk', query_embs, query_mask, cand_embs, cand_mask)
        sims = torch.einsum('bld,ckd->bclk', query_embs, cand_embs)
        mask = torch.einsum('bl,ck->bclk', query_mask, cand_mask)
        sims[mask == 0] = sims.min() - 1e5
    else:
        if query_mask is None:
            query_mask = torch.ones(query_embs.shape[0], device=query_embs.device)
        # sims = torch.einsum('ld,l,ckd,ck->clk', query_embs, query_mask, cand_embs, cand_mask)
        sims = torch.einsum('ld,ckd->clk', query_embs, cand_embs)
        mask = torch.einsum('l,ck->clk', query_mask, cand_mask)
        sims[mask == 0] = sims.min() - 1e5
    return sims

def sims_to_token_recalls(sims, query_idf):
    # sims: bclk or clk
    # query_idf: bl
    recall = sims.max(axis=-1).values   # bclk -> bcl or clk -> cl
    recall_scale = query_idf / query_idf.sum(axis=-1, keepdims=True)  # bl / b1 -> bl or l / 1 -> l
    recall_scale = recall_scale.unsqueeze(-2)    # b1l or 1l
    recall = recall * recall_scale # bcl * b1l -> bcl or cl * 1l -> cl
    return recall

def sims_to_token_precision(sims, cand_idf):
    # sims: bclk or clk
    # cand_idf: ck
    precision = sims.max(axis=-2).values    # bclk -> bck or clk -> ck
    precision_scale = cand_idf / cand_idf.sum(axis=-1, keepdims=True)    # ck / c1 -> ck or k / 1 -> k
    precision = precision * precision_scale # bck * ck -> bck or ck * k -> ck
    return precision

def sims_to_bertscore(sims, query_idf, cand_idf, metric='recall'):
    # sims: bclk or clk
    # query_idf: bl or l
    # cand_idf: ck
    if metric in ['recall', 'f1']:
        token_recalls = sims_to_token_recalls(sims, query_idf)    # bcl or cl
        R = token_recalls.sum(axis=-1) # bc or c
        if metric == 'recall':
            return R
    if metric in ['precision', 'f1']:
        token_precisions = sims_to_token_precision(sims, cand_idf)    # bck or ck
        P = token_precisions.sum(axis=-1)   # bc or c
        if metric == 'precision':
            return P
    return 2 * R * P / (R + P)

def pad_embs_idfs(emb, idf, device):
    import torch
    from torch.nn.utils.rnn import pad_sequence

    emb = [torch.from_numpy(e) for e in emb]
    idf = [torch.from_numpy(i) for i in idf]
    lens = [e.size(0) for e in emb]
    emb_pad = pad_sequence(emb, batch_first=True, padding_value=2.0)
    idf_pad = pad_sequence(idf, batch_first=True)

    def length_to_mask(lens):
        lens = torch.tensor(lens, dtype=torch.long)
        max_len = max(lens)
        base = torch.arange(max_len, dtype=torch.long).expand(len(lens), max_len)
        return base < lens.unsqueeze(1)

    pad_mask = length_to_mask(lens)
    return emb_pad.to(device), pad_mask.to(device), idf_pad.to(device)

@attr.s(auto_attribs=True)
class BertScoreSelectorArgs(CommonSelectorArgs):
    emb_lm: str = 'microsoft/deberta-base-mnli'
    metric: str = 'recall'
    idf: bool = False
    ordering: str | None = None
    coverage: bool = True
    add_cand_score: bool | None = False
    cand_score_discount: float | None = 1

    def get_name(self) -> str:
        name_parts = [self.emb_lm.split('/')[-1], self.metric]
        if self.idf: name_parts.append('idf')
        if self.coverage: name_parts.append('coverage')
        if self.ordering: name_parts.append(f'{self.ordering}_order')
        if self.add_cand_score:
            if self.cand_score_discount != 1:
                name_parts.append(f'candscore_by{self.cand_score_discount}')
            else:
                name_parts.append('candscore')
        return '-'.join(name_parts)

    def get_friendly_name(self):
        name_parts = [self.emb_lm.split('/')[-1], self.metric]
        if self.idf: name_parts.append('idf')
        if self.coverage: name_parts.append('coverage')
        if self.ordering: name_parts.append(f'{self.ordering}_order')
        if self.add_cand_score:
            if self.cand_score_discount != 1:
                name_parts.append(f'candscore_by{self.cand_score_discount}')
            else:
                name_parts.append('candscore')
        return '+'.join(name_parts)


class BertScoreSelector(BaseExampleSelector, SelectorUtilsMixin, BaseModel):
    args: BertScoreSelectorArgs
    example_template: ExampleTemplate
    demo_candidates: Dataset
    query2idx: dict[str, int]

    scores: np.ndarray = None
    cand_embs: torch.Tensor = None
    cand_mask: torch.Tensor = None
    cand_idfs: torch.Tensor = None
    query_embs: torch.Tensor = None
    query_mask: torch.Tensor = None
    query_idfs: torch.Tensor = None

    shot_scores_l: np.ndarray | list[np.ndarray] | None = None
    shot_idxs_l: np.ndarray | list[np.ndarray] | None = None
    # cands_argsort_l: np.ndarray = None

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True


    def add_example(self, example: dict[str, str]) -> Any:
        ...

    @staticmethod
    def get_covering_shot_idxs(
        args: BertScoreSelectorArgs,
        query_embs, query_idfs, cand_embs_l, cand_mask_l, cand_idfs_l,
        cand_lens=None, max_len=-1, return_scores=False,
    ):
        assert args.metric == 'recall'
        with torch.no_grad():
            n_shots = args.n_shots
            sims_l = []
            token_recalls_l = []
            for _cand_embs, _cand_mask, _cand_idfs in zip(cand_embs_l, cand_mask_l, cand_idfs_l):
                _cand_embs = _cand_embs.to(query_embs.device)
                _cand_mask = _cand_mask.to(query_embs.device)
                _sims = compute_sims(query_embs, None, _cand_embs, _cand_mask)
                _token_recalls = sims_to_token_recalls(_sims, query_idfs).cpu().numpy()
                sims_l.append(_sims)
                token_recalls_l.append(_token_recalls)
            token_recalls = np.concatenate(token_recalls_l, axis=0)
            shot_idxs, stats = decomposed_coverage_greedy(
                n_shots, token_recalls, args.add_cand_score, args.cand_score_discount,
                cand_lens=cand_lens, max_len=max_len)
            shot_scores = token_recalls[shot_idxs].sum(axis=-1)
            if args.ordering:
                if args.ordering != 'recall':
                    raise NotImplementedError
                    def lls(ll, idxes, batch_size):
                        sel = []
                        for i in idxes:
                            sel.append(ll[i//batch_size][i%batch_size])
                    shot_scores = sims_to_bertscore(
                        sims[shot_idxs], query_idfs, cand_idfs[shot_idxs],
                        metric=args.ordering)
                order = shot_scores.argsort()
                shot_idxs = np.array(shot_idxs)[order]
                shot_scores = shot_scores[order]
        if return_scores:
            return shot_idxs, shot_scores
        else:
            return shot_idxs

    def select_examples(self, input_variables: dict[str, str], return_scores=False) -> list[dict]:
        query = self.example_template.format(**input_variables, embedding=True)
        if query not in self.query2idx:
            if self.args.coverage:
                shot_idxs = self.get_covering_shot_idxs(
                    self.args,
                    self.query_embs[self.query2idx[query]],
                    self.query_mask[self.query2idx[query]],
                    self.query_idfs[self.query2idx[query]],
                    self.cand_embs, self.cand_mask, self.cand_idfs)
            else:
                scores = self.scores[self.query2idx[query]]
                shot_idxs = scores.argsort()[-self.args.n_shots:]
                shot_scores = scores[shot_idxs]
        else:
            shot_idxs = self.shot_idxs_l[self.query2idx[query]]
            shot_scores = self.shot_scores_l[self.query2idx[query]]

        if return_scores:
            return self.demo_candidates.select(shot_idxs), shot_scores
        else:
            return self.demo_candidates.select(shot_idxs)

    @classmethod
    def from_examples(
        cls,
        args: BertScoreSelectorArgs,
        examples: list[dict],
        example_template: ExampleTemplate,
        query_examples: list[dict] = None,
        enc_len_fn: Any = None,
        max_len: int = -1,
        subtract_gen_len: bool = False,
        device: str = 'cpu',
        progress_bar: bool = True,
    ) -> BertScoreSelector:
        import torch
        examples = cls.drop_duplicates(examples, example_template)
        ex_to_string = lambda ex: example_template.format(**ex, embedding=True)
        query_examples = query_examples or []

        cand_strings = [ex_to_string(ex) for ex in examples]
        query_strings = [ex_to_string(ex) for ex in query_examples]
        query2idx = {query: i for i, query in enumerate(query_strings)}
        max_len = max_len if max_len > 0 else 1000000
        cand_lens = [enc_len_fn(example_template.format(**ex)) if enc_len_fn else 0
                    for ex in examples]
        query_lens = [enc_len_fn(example_template.format(**ex, test=True)) if enc_len_fn else 0
                    for ex in query_examples]
        completed_query_lens = [enc_len_fn(example_template.format(**ex)) if enc_len_fn else 0
                    for ex in query_examples]


        cand_contexts, query_contexts = [''] * len(examples), [''] * len(query_examples)
        if hasattr(example_template, 'embed_context') and example_template.embed_context:
            cand_contexts = [cand_str[:-len(example_template.get_source(ex))]
                             for cand_str, ex in zip(cand_strings, examples)]
            query_contexts = [query_str[:-len(example_template.get_source(ex))]
                             for query_str, ex in zip(query_strings, examples)]
        n_queries = len(query_examples)

        tokenizer = get_tokenizer(args.emb_lm, use_fast=False)
        model = get_model(args.emb_lm, model2layers[args.emb_lm]).eval()
        model = model.to(device)
        from bert_score.utils import get_idf_dict, sent_encode, get_bert_embedding
        if not args.idf:
            idf_dict = defaultdict(lambda: 1.0)
            # set idf for [SEP] and [CLS] to 0
            idf_dict[tokenizer.sep_token_id] = 0
            idf_dict[tokenizer.cls_token_id] = 0
        else:
            idf_dict = get_idf_dict(cand_strings, tokenizer)

        ls = lambda l, idxes: [l[i] for i in idxes]

        with torch.no_grad():
            cand_embs, cand_idfs = embed(cand_strings, model, tokenizer, idf_dict, device, cand_contexts)
            query_embs, query_idfs = embed(query_strings, model, tokenizer, idf_dict, device, query_contexts)
            model = model.to('cpu')
            del model
            torch.cuda.empty_cache()

            def make_chunks(lens, max_prod):
                i = 0
                chunks = [[]]
                while i < len(lens):
                    cand_chunk = chunks[-1] + [i]
                    if max([lens[j] for j in cand_chunk]) * len(cand_chunk) > max_prod:
                        chunks.append([])
                        continue
                    chunks[-1].append(i)
                    i += 1
                return chunks
            cand_embs_l, cand_mask_l, cand_idfs_l = [], [], []
            # breakpoint()
            for c_idxes in track(make_chunks([emb.shape[0] for emb in cand_embs], max_prod=1000000)):
            # for c_idxes in chunked(range(len(cand_embs)), 5000):
                _cand_embs = ls(cand_embs, c_idxes)
                _cand_idfs = ls(cand_idfs, c_idxes)
                _cand_embs, _cand_mask, _cand_idfs = pad_embs_idfs(_cand_embs, _cand_idfs, 'cpu')
                cand_embs_l.append(_cand_embs)
                cand_mask_l.append(_cand_mask)
                cand_idfs_l.append(_cand_idfs)
            # breakpoint()

        if not args.coverage:
            batch_size = 1
            query_iter = track(chunked(range(n_queries), batch_size), description='Finding shots', total=np.ceil(n_queries/batch_size)) if progress_bar else range(0, n_queries, batch_size)
            with torch.no_grad():
                def get_batch_scores(q_idxes):
                    _query_embs = ls(query_embs, q_idxes)
                    _query_idfs = ls(query_idfs, q_idxes)
                    # TODO: try padding queries a priori
                    _query_embs, _query_mask, _query_idfs = pad_embs_idfs(_query_embs, _query_idfs, device)
                    scores_l = []
                    for _cand_embs, _cand_mask, _cand_idfs in zip(cand_embs_l, cand_mask_l, cand_idfs_l):
                        _cand_embs = _cand_embs.to(device)
                        _cand_mask = _cand_mask.to(device)
                        _cand_idfs = _cand_idfs.to(device)
                        sims = compute_sims(_query_embs, _query_mask, _cand_embs, _cand_mask)
                        scores = sims_to_bertscore(
                            sims, _query_idfs, _cand_idfs, metric=args.metric)
                        scores_l.append(scores)
                        # torch.cuda.empty_cache()
                    scores = torch.cat(scores_l, axis=1)
                    return scores
                batch_scores = []
                for q_idxes in query_iter:
                    batch_scores.append(get_batch_scores(q_idxes).cpu())
                scores = torch.cat(batch_scores, axis=0)
                shot_scores_l = scores.sort(axis=-1).values[:, -args.n_shots:].numpy()
                shot_idxs_l = scores.argsort(axis=-1)[:, -args.n_shots:].numpy()
            torch.cuda.empty_cache()
            return cls(
               args=args,
                example_template=example_template,
                demo_candidates=examples,
                query2idx=query2idx,
                # scores=scores,
                shot_scores_l=shot_scores_l,
                shot_idxs_l=shot_idxs_l,
            )
        else:
            query_iter = track(range(n_queries), description='Finding shots', total=n_queries) if progress_bar else range(n_queries)
            shot_idxs_l, shot_scores_l = [], []
            for idx in query_iter:
                if not subtract_gen_len:
                    _max_len = max_len - query_lens[idx]
                else:
                    _max_len = max_len - completed_query_lens[idx] - 4
                _query_embs = torch.from_numpy(query_embs[idx]).to(device)
                _query_idfs = torch.from_numpy(query_idfs[idx]).to(device)
                shot_idxs, shot_scores = cls.get_covering_shot_idxs(
                    args, _query_embs, _query_idfs, cand_embs_l, cand_mask_l, cand_idfs_l, cand_lens, _max_len, return_scores=True)
                shot_idxs_l.append(np.array(shot_idxs))
                shot_scores_l.append(np.array(shot_scores))
            print(f'Average number of shots: {np.mean([len(shot_idxs) for shot_idxs in shot_idxs_l])}')

            torch.cuda.empty_cache()
            return cls(
                args=args,
                example_template=example_template,
                demo_candidates=examples,
                query2idx=query2idx,
                shot_scores_l=shot_scores_l,
                shot_idxs_l=shot_idxs_l
            )

def test():
    import numpy as np
    from functools import partial
    from pathlib import Path
    from driver import get_dataset, get_templates
    from constants import Dataset as DS
    from tools.track import track
    from bert_score import BERTScorer

    # dataset, input_feature, test_split = DS.SMCALFLOW_CS, 'paraphrase', 'test'
    # dataset, input_feature, train_split, test_split = DS.GEOQUERY, 'source', 'template_1_train', 'template_1_test'
    # dataset, input_feature, train_split, test_split = DS.OVERNIGHT, 'paraphrase', 'socialnetwork_template_0_train', 'socialnetwork_template_0_test'
    # dataset, input_feature, test_split = DS.BREAK, 'question_text', 'validation'
    dataset, input_feature, train_split, test_split = DS.QNLI, None, 'train', 'validation'
    ds = get_dataset(dataset, data_root=Path('../data'))
    candidates = ds[train_split].select(list(range(min(500, len(ds[train_split])))))
    templates = get_templates(dataset, 'v1', input_feature=input_feature)
    example_template = templates['example_template']

    args = BertScoreSelectorArgs(coverage=False)
    bert_scorer = BERTScorer(lang='en', model_type=args.emb_lm, idf=args.idf, device='cuda:1')
    bs_selector = BertScoreSelector.from_examples(args, candidates, example_template, query_examples=ds[test_split], device=0)
    demo_cands = bs_selector.demo_candidates
    n_cands = len(demo_cands)
    scores0 = bert_scorer.score(demo_cands['source'], ds[test_split]['source'][:1] * n_cands)
    scores1 = bs_selector.scores
    assert np.allclose(scores0[1].numpy(), scores1[0, :])

    args = BertScoreSelectorArgs(coverage=False, idf=True)
    bs_selector = BertScoreSelector.from_examples(args, candidates, example_template, query_examples=ds[test_split], device=0)
    demo_cands = bs_selector.demo_candidates
    n_cands = len(demo_cands)
    bert_scorer = BERTScorer(lang='en', model_type=args.emb_lm, idf=args.idf, idf_sents=demo_cands['source'], device='cuda:1')
    scores0 = bert_scorer.score(demo_cands['source'], ds[test_split]['source'][:1] * n_cands)
    scores1 = bs_selector.scores
    assert np.allclose(scores0[1].numpy(), scores1[0, :])
