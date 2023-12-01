from itertools import repeat
from torch.multiprocessing import Pool, set_start_method
import math

import torch
import numpy as np
from torch import Tensor
from entmax import entmax_bisect

from multi_kmeans import MultiKmeans
from utils import get_length_mask, dot_product_and_mask, subsequent_mask, unsqueeze_as, neighbours_mask
from stats import compute_sparsity, compute_recall, compute_exact_fraction


def quantize(x: Tensor,
             bucket_size: int,
             lengths: Tensor,
             enforce_equal_size: bool = False,
             return_lower_bounds: bool = False,
             temperature: float = 1.):
    """
    Args:
        x: projected query or key (batch, heads, n, projections)
        bucket_size: number of tokens per bucket
        lengths: tensor (batch,) - necessary for masking padding
        enforce_equal_size: makes all buckets with the same number of tokens.
            If not, buckets will have tokens falling within the same interval
        return_lower_bounds: return also a tensor with the lowest value in each
            bucket
        temperature: coefficient to multiply projected values before tanh
            (only used with variable-sized buckets)

    Return:
        (buckets, inds, lower_bounds[Optional]):
            buckets (batch, heads, n, projections): the bucket each token has
                been assigned to at each projection
            lower_bounds [Optional] (num_intervals, rounds): the lowest value
                assigned to each bucket
    """
    batch_size, num_heads, n, num_projections = x.shape

    mask = get_length_mask(lengths, n)
    x = x.masked_fill(~mask.view(batch_size, 1, n, 1), np.inf)
    num_intervals = math.ceil(n / bucket_size)

    if enforce_equal_size:
        modulo = n % bucket_size
        if modulo != 0:
            # pad
            shape = [batch_size, num_heads, bucket_size - modulo,
                     num_projections]
            padding = torch.full(shape, np.inf).to(x.device)
            x = torch.cat([x, padding], dim=2)

        # each round will have all tokens sorted by value
        inds = x.argsort(2)
        rev_inds = inds.argsort(2)
        buckets = rev_inds // bucket_size

        # remove the padding needed for having a multiple of `intervals`
        buckets = buckets[:, :, :n]

    else:
        step = 2 / num_intervals
        num_boundaries = num_intervals - 1
        boundaries = torch.linspace(-1 + step, 1 - step, num_boundaries)
        boundaries = boundaries.to(x.device)
        # torch.bucketize fails on tensors requiring gradient
        x = torch.tanh(temperature * x.detach())
        buckets = torch.bucketize(x, boundaries)

    if return_lower_bounds:
        raise NotImplementedError

    return buckets


def group_by_buckets(
        qproj,
        kproj,
        bucket_size,
        lengths,
        enforce_same_bounds=False,
        enforce_same_size=True,
        temperature=1.,
        clusters_per_head=None,
        top_clusters=1
):
    """
    Args:
        kproj: projections of keys (batch, heads, n, projections)
        qproj: same as kproj
        bucket_size: number of tokens per bucket
        lengths: length of each item (batch,); necessary for masking padding
        enforce_same_bounds: in case of quantization, enforce queries to use
            the same lower bounds than keys. This will allow a different number
            of queries per bucket.
        enforce_same_size: whether buckets must have the same size (same number
            of queries and keys). If False, some buckets may have few or no
            elements.
        temperature: coefficient applied before tanh
        clusters_per_head: use clustering (e.g. kmeans) instead of quantization
        top_clusters: how many clusters to consider as hashing rounds

    Returns:
        buckets_q, buckets_k: tuple of tensors (batch, heads, n, projections)
    """
    if enforce_same_bounds:
        buckets_k, inds_k, lower_bounds = quantize(
            kproj, bucket_size, lengths, True, True, temperature)
        buckets_q = torch.zeros_like(buckets_k)
        lower = torch.full_like(lower_bounds[0], float('-inf'))

        for bucket, upper in enumerate(lower_bounds[1:]):
            # skip the first row - lower values go to bucket 0 anyway
            upper = upper.view(1, -1)
            inds = torch.logical_and(lower < qproj, qproj <= upper)
            buckets_q[inds] = bucket
            lower = upper

        # values above the highest lower bound go to the last bucket
        last_bucket = math.ceil(qproj.shape[2] / bucket_size) - 1
        buckets_q[qproj > lower] = last_bucket
    elif clusters_per_head is not None:
        # lengths ignored - padded tokens are computed, will be discarded later outside this function
        buckets_q = predict_clusters(qproj, clusters_per_head, top_clusters=top_clusters)
        buckets_k = predict_clusters(kproj, clusters_per_head, top_clusters=top_clusters)
    else:
        buckets_k = quantize(
            kproj, bucket_size, lengths, enforce_same_size,
            temperature=temperature)
        buckets_q = quantize(
            qproj, bucket_size, lengths, enforce_same_size,
            temperature=temperature)

    return buckets_q, buckets_k


def group_by_distance(
        qproj,
        kproj,
        lengths,
        gold_probas,
        dist='cos',
        threshold=0.5,
        p=2.0,
        window_inds=None,
        num_positive=None,
        causal=False,
        add_cls_mask=False,
        add_last_mask=False,
        layer_id=0,
        clusters_per_head=None,
        top_clusters=0
):
    """
    Group q and k vectors if dist(q, k) < threshold.
    If window inds are specified, add windowed tokens to the group.
    """
    if dist == 'cos':
        qproj_norm = qproj.norm(p=2, dim=-1).unsqueeze(-1)
        kproj_norm = kproj.norm(p=2, dim=-1).unsqueeze(-1)
        cos_sim = (qproj / qproj_norm) @ (kproj / kproj_norm).transpose(-1, -2)
        pairwise_ds = 1 - cos_sim
    elif dist == 'euclidean':
        pairwise_ds = torch.cdist(qproj, kproj, p=2.0)
    elif dist == 'l2':
        pairwise_ds = torch.cdist(qproj, kproj, p=2.0) ** 2
    elif dist == 'direction':
        qproj_norm = qproj.norm(p=p, dim=-1).unsqueeze(-1).clamp(min=1e-7)
        kproj_norm = kproj.norm(p=p, dim=-1).unsqueeze(-1).clamp(min=1e-7)
        pairwise_ds = torch.cdist(qproj / qproj_norm, kproj / kproj_norm, p=p)
    elif dist == 'random':
        bs, nh, slen, _ = qproj.shape
        pairwise_ds = torch.rand(bs, nh, slen, slen, device=qproj.device)
    elif dist == 'dotproduct_entmax':
        dots = dot_product_and_mask(qproj, kproj, lengths)
        att_dist = entmax_bisect(dots, alpha=threshold[0], dim=-1)
        pairwise_ds = (att_dist == 0).float() * 999.0 + (att_dist > 0).float() * 0.0
    elif dist == 'min':
        qk_diff = qproj.unsqueeze(-2) - kproj.unsqueeze(-3)
        pairwise_ds, _ = torch.min(qk_diff**2, dim=-1)
    elif dist == 'min_norm':
        qproj_norm = qproj.norm(p=p, dim=-1).unsqueeze(-1).clamp(min=1e-7)
        kproj_norm = kproj.norm(p=p, dim=-1).unsqueeze(-1).clamp(min=1e-7)
        qproj_unit = qproj / qproj_norm
        kproj_unit = kproj / kproj_norm
        qk_diff = qproj_unit.unsqueeze(-2) - kproj_unit.unsqueeze(-3)
        pairwise_ds, _ = torch.min(qk_diff**2, dim=-1)
    elif dist == 'alt_min':
        raise NotImplementedError
        qproj_norm = qproj.norm(p=2, dim=-1).unsqueeze(-1)
        kproj_norm = kproj.norm(p=2, dim=-1).unsqueeze(-1)
        cos_sim = ((qproj / qproj_norm).unsqueeze(-1) * (kproj / kproj_norm).transpose(-1, -2).unsqueeze(-3)).min(-2)  # .sum(-2) is equal to dot-prod
        pairwise_ds = 1 - cos_sim
    elif dist == 'minkowski':
        # minkowski distance:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
        pairwise_ds = torch.cdist(qproj, kproj, p=p)
    else:
        # 'bigbird', 'reformer', 'reformer_rounds', 'longformer', 'openai_sparse', 'routing'
        bs, nh, slen, _ = qproj.shape
        pairwise_ds = torch.rand(bs, nh, slen, slen, device=qproj.device)

    max_length = qproj.shape[-2]
    num_heads = qproj.shape[1]
    assert len(threshold) == 1 or len(threshold) == num_heads
    expanded_threshold = torch.tensor(threshold). \
        reshape(1, len(threshold), 1, 1).to(qproj.device)
    pairwise_ds_thresholded = pairwise_ds < expanded_threshold

    if dist == 'bigbird':
        from bigbird import bigbird_simulated_attention
        bs, nh, slen, _ = qproj.shape
        # win_size = 1 if window_inds is None else window_inds.shape[1]
        num_rand_blocks = int(threshold[0])
        mask = get_length_mask(lengths, max_length)
        pairwise_mask = mask.unsqueeze(-1) & mask.unsqueeze(1)
        pairwise_mask.unsqueeze_(1)
        x = bigbird_simulated_attention(
            pairwise_mask.cpu().numpy(),
            num_attention_heads=nh,
            num_rand_blocks=num_rand_blocks,  # default from bigbird repo is 3
            from_seq_length=slen,
            to_seq_length=slen,
            from_block_size=1,
            to_block_size=1,
            max_seq_len=slen
        )
        pairwise_ds_thresholded = x.to(qproj.device).bool()
    elif dist == 'longformer':
        from longformer import longformer_simulated_attention
        max_globals = int(threshold[0])
        mask = get_length_mask(lengths, max_length)
        sim_mask = longformer_simulated_attention(
            mask,
            window_size=1,
            dilation=0,
            max_globals_per_sample=max_globals,
            min_globals_per_sample=2,
        )
        pairwise_ds_thresholded = sim_mask.bool().unsqueeze(1)
    elif dist == 'openai_sparse':
        from openai_sparse import openai_sparse_simulated_attention
        stride = int(threshold[0])
        mask = get_length_mask(lengths, max_length)
        sim_mask = openai_sparse_simulated_attention(
            mask,
            qproj.shape[-1],
            qproj.shape[1],
            stride=stride,
            expressivity=2,
            is_bidirectional=not causal
        )
        pairwise_ds_thresholded = sim_mask.bool().unsqueeze(0).unsqueeze(1).repeat(qproj.shape[0], 1, 1, 1)
    elif dist == 'reformer' or dist == 'reformer_rounds':
        from reformer import reformer_simulated_attention
        # https://github.com/lucidrains/reformer-pytorch/blob/master/reformer_pytorch/reformer_pytorch.py#L522
        num_buckets = int(threshold[0])
        num_hashes = qproj.shape[-1] if dist == 'reformer_rounds' else 1
        qk = torch.cat([qproj, kproj], dim=-1)
        mask = get_length_mask(lengths, max_length)
        sim_mask = reformer_simulated_attention(
            qk,
            lsh_attn_chunk_length=None,  # default from reformer is 4 or 8 for longer seqs
            num_buckets=num_buckets,
            num_hashes=num_hashes,  # if > 1, use any(-1) to represent matches
            mask=mask
        )
        pairwise_ds_thresholded = sim_mask.bool()
    elif dist == 'routing' or dist == 'routing_trained' or dist == 'routing_extender':
        from routing_transformer import routing_simulated_attention
        topk_window_size = None if top_clusters == 0 else top_clusters
        sim_mask = routing_simulated_attention(qproj, kproj, clusters_per_head, topk_window_size=topk_window_size)
        pairwise_ds_thresholded = sim_mask.bool().to(qproj.device)

    if window_inds is not None:
        window_size = window_inds.shape[-1]
        win_mask = neighbours_mask(max_length, window_size).unsqueeze(0).unsqueeze(1).to(qproj.device).bool()
        pairwise_ds_thresholded |= win_mask
        # r = torch.arange(max_length).view(-1, 1)
        # pairwise_ds_thresholded[:, :, r, window_inds] = True

    if add_cls_mask is True:
        pairwise_ds_thresholded[:, :, 0, :] = True  # from cls
        pairwise_ds_thresholded[:, :, :, 0] = True  # to cls

    if add_last_mask is True:
        pairwise_ds_thresholded[:, :, -1, :] = True  # from last
        pairwise_ds_thresholded[:, :, :, -1] = True  # to last

    # pairwise_mask is (batch, heads, n, n)
    pad_mask = get_length_mask(lengths, max_length)
    if causal:
        causal_mask = subsequent_mask(max_length).bool().to(pad_mask.device)
        pairwise_mask = pad_mask.unsqueeze(-1) & pad_mask.unsqueeze(1)
        pairwise_mask = pairwise_mask & causal_mask.unsqueeze(0)
    else:
        causal_mask = None
        pairwise_mask = pad_mask.unsqueeze(-1) & pad_mask.unsqueeze(1)

    # fix some corners cases
    if pairwise_ds_thresholded.shape[0] == 1:
        bs = qproj.shape[0]
        pairwise_ds_thresholded = pairwise_ds_thresholded.repeat(bs, 1, 1, 1)
    if pairwise_ds_thresholded.shape[1] == 1:
        num_heads = qproj.shape[1]
        pairwise_ds_thresholded = pairwise_ds_thresholded.repeat(1, num_heads, 1, 1)

    sparsity_per_head = compute_sparsity(pairwise_ds_thresholded.float(), pad_mask, causal_mask=causal_mask)
    recall_per_head = compute_recall(
        pairwise_ds_thresholded.float(), gold_probas.float(), pad_mask, causal_mask=causal_mask)
    exact_per_head = compute_exact_fraction(
        pairwise_ds_thresholded.float(), gold_probas.float(), pad_mask, causal_mask=causal_mask)

    pairwise_ds_thresholded = pairwise_ds_thresholded.masked_fill(~pairwise_mask.unsqueeze(1), False)
    return sparsity_per_head, recall_per_head, exact_per_head, pairwise_ds_thresholded.float()


def learn_clusters_for_head(n_clusters, rounds, data_head):
    kmeans = MultiKmeans(n_clusters=n_clusters, rounds=rounds).fit(data_head)
    return kmeans.fit(data_head)


def learn_clusters(dataset, layer, proj_q, proj_k, n_clusters=16, rounds=1, concat_q_and_k=False):
    data = dataset.get_clustering_data(layer, proj_q, proj_k, dataset="train", concat_q_and_k=concat_q_and_k)
    heads, elems, dim = data.shape
    kmeans_heads = []
    for h in range(heads):
        kmeans = learn_clusters_for_head(n_clusters, rounds, data[h])
        kmeans_heads.append(kmeans)
    return kmeans_heads


def predict_clusters_for_head(cluster, x_head, top_clusters=1):
    batch_size, seq_len, dim = x_head.shape
    device = x_head.device
    x_head = x_head.reshape(batch_size*seq_len, dim).cpu().detach().numpy()
    preds = cluster.predict(x_head, top_clusters=top_clusters)
    # shape: (rounds, batch*seq_len) if top_clusters==1 else (rounds*k, batch*seq_len)
    preds = torch.from_numpy(preds).to(device)
    adjust_rounds = cluster.kmeans[0].n_clusters if top_clusters > 1 else 1
    preds = preds.transpose(1, 0).reshape(batch_size, seq_len, cluster.rounds * adjust_rounds)
    return preds


def predict_clusters(x, clusters_per_head, top_clusters=1):
    if isinstance(clusters_per_head, torch.Tensor):
        bsz = x.shape[0]
        expanded_centroids = clusters_per_head.unsqueeze(0).expand(bsz, -1, -1, -1, -1).to(x.device)
        expanded_x_low = x.unsqueeze(2)
        x_diffs_sq = (expanded_x_low.unsqueeze(-2).double() - expanded_centroids.unsqueeze(-3).double())**2
        x_dists = torch.sum(x_diffs_sq, dim=-1)
        _, x_clusters = torch.topk(x_dists, top_clusters, dim=-1, largest=False)
        x_clusters = x_clusters.squeeze(2)
    else:
        x_clusters = []
        for head in range(len(clusters_per_head)):
            cluster = clusters_per_head[head]
            x_head = x[:, head, :, :]
            preds = predict_clusters_for_head(cluster, x_head, top_clusters=top_clusters)
            x_clusters.append(preds)
        x_clusters = torch.stack(x_clusters).transpose(0, 1)
    return x_clusters
