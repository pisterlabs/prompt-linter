import custom_types
import numpy.typing as npt
import langchain
import copy
import asyncio
import numpy as np
import numbers
from typing import List, Tuple
from langchain.llms.base import LLM
import cluster
import json
import os
import sys
sys.path.append(os.getcwd())
from model_wrapper import llm, llm_name

# STOP_DEFAULT_TOKENS = ["### Instruction", "\n"]
STOP_DEFAULT_TOKENS = []


def get_theorems_in_group(embeddings: custom_types.Embeddings, labels: npt.NDArray, group_idx: int, max_size=None, random=True):
    s = [embeddings[i][0] for i in np.where(labels == group_idx)[0]]
    if max_size is None or len(s) <= max_size:
        return s
    if not random:
        return s[:max_size]
    c = np.random.choice(np.arange(len(s)), size=max_size, replace=False)
    return [s[i] for i in c]


# From https://github.com/hichamjanati/pyldpc, but modified to make a square matrix
def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def parity_check_matrix(n_code, d_v, d_c, seed=None):
    """
    Build a regular Parity-Check Matrix H following Callager's algorithm.

    Parameters
    ----------
    n_code: int, Length of the codewords.
    d_v: int, Number of parity-check equations including a certain bit.
        Must be greater or equal to 2.
    d_c: int, Number of bits in the same parity-check equation. d_c Must be
        greater or equal to d_v and must divide n.
    seed: int, seed of the random generator.

    Returns
    -------
    H: array (n_equations, n_code). LDPC regular matrix H.
        Where n_equations = d_v * n / d_c, the total number of parity-check
        equations.

    """
    rng = check_random_state(seed)

    if d_v <= 1:
        raise ValueError("""d_v must be at least 2.""")

    # if d_c <= d_v:
    #     raise ValueError("""d_c must be greater than d_v.""")

    if n_code % d_c:
        raise ValueError("""d_c must divide n for a regular LDPC matrix H.""")

    n_equations = (n_code * d_v) // d_c

    block = np.zeros((n_equations // d_v, n_code), dtype=int)
    H = np.empty((n_equations, n_code))
    block_size = n_equations // d_v

    # Filling the first block with consecutive ones in each row of the block

    for i in range(block_size):
        for j in range(i * d_c, (i+1) * d_c):
            block[i, j] = 1
    H[:block_size] = block

    # reate remaining blocks by permutations of the first block's columns:
    for i in range(1, d_v):
        H[i * block_size: (i + 1) * block_size] = rng.permutation(block.T).T
    H = H.astype(int)
    return H


parity_check_matrix(16, 4, 4)


async def local_neighbor_with_descr_labels(thms_node: List[str], descr_node: str, thms_local: List[List[str]], descr_thms_local: List[str], llm: LLM):
    description_exists = len(descr_node) != 0

    merged_non_prim = [f"Description: {descr_thms_local[i]}\n" + "\n".join(t) + "\n" for i, t in enumerate(thms_local)] if description_exists \
        else ["\n".join(t) for t in thms_local]
    joined_non_prim = "\n\n".join(merged_non_prim)

    joined_prim = (f"Description: {descr_node}" +
                   "\n" if descr_node != "" else "") + "\n".join(thms_node)
    # TODO: Fix up so that we can use any format of promtp
    # TODO: hmmmm instruction???
    if not description_exists:
        prompt = f"""You will be given a set of primary theorems and a set of non primary theorems. Can you give a brief textual and qualitative description of the unifying theme behind the primary theorems?

NON-PRIMARY THEOREMS: "{joined_non_prim}"

PRIMARY THEOREMS: "{joined_prim}"

SHORT RESPONSE:
"""
    else:
        prompt = f"""You will be given a set of primary theorems and a set of non primary theorems as well as their current description. Can you give a brief and updated textual and qualitative description of the unifying theme behind the primary theorems?

Non-primary theorems: "{joined_non_prim}"

Primary theorems: "{joined_prim}"

SHORT RESPONSE:
""" 
    # print("Prompt", prompt)
    try:
        r = await llm.agenerate([prompt])
        # r = await llm.agenerate([prompt], stop=STOP_DEFAULT_TOKENS)
    except:
        print("ERROR GENERATING for prompt", prompt)
        print("RETURNING ORIGINAL DESCR")
        return descr_node
    finally:
        pass
    print("Generated", r)
    return r.generations[0][0].text


class RunParams:
    n_clusters: int
    seed: int
    n_rounds: int
    model_name: str
    max_sample_size: int
    descr: str
    cluster_cluster_deg: int

    def __init__(self, n_clusters: int, seed: int, n_rounds: int, model_name: str, max_sample_size: int, cluster_cluster_deg=3, descr: str = "default"):
        self.n_clusters = n_clusters
        self.seed = seed
        self.n_rounds = n_rounds
        self.model_name = model_name
        self.max_sample_size = max_sample_size
        self.descr = descr
        self.cluster_cluster_deg = cluster_cluster_deg

    def to_dict(self):
        return self.__dict__


class RunData:
    # The outer list for each round. The middle list is the list of messages per node, the inner list is the specific messages to its neighbor
    rounds: List[
        List[List[str]]
    ] = []
    parity_check_matrix: npt.NDArray = None
    params: RunParams
    completed_rounds = 0
    cluster_labels: npt.NDArray
    shortened: List[Tuple[str, List[str]]] = []

    def __init__(self, cluster_labels: npt.NDArray, parity_check_matrix: npt.NDArray, params: RunParams) -> None:
        self.cluster_labels = cluster_labels
        self.parity_check_matrix = parity_check_matrix
        self.params = params

    def to_dict(self):
        return {
            "rounds": self.rounds,
            "parity_check_matrix": np.array(self.parity_check_matrix).tolist(),
            "params": self.params.to_dict(),
            "completed_rounds": self.completed_rounds,
            "cluster_labels": np.array(self.cluster_labels).tolist(),
            "shortened": self.shortened
        }

    def from_dict(d: dict):
        r = RunData(np.array(d["cluster_labels"]), np.array(
            d["parity_check_matrix"]), RunParams(**d["params"]))
        r.rounds = d["rounds"]
        r.completed_rounds = d["completed_rounds"]
        return r


def get_data_file_name(params: RunParams):
    return f"data_store/llm_bp_clustersize_{params.n_clusters}__seed_{params.seed}_{params.model_name}__descr_{params.descr}.json"


def save_dict(params: RunParams, d: RunData):
    json.dump(d.to_dict(), open(get_data_file_name(params), "w"))


async def llm_bp(embeddings: custom_types.Embeddings, llm: LLM, data: RunData):
    """
      Run bp-ish.... TODO: document

      Because K-clustering initializes the clusters randomly, we can assume that we each cluster is distinct from each other (i.e. cluster 1 and 2 are not correlated any differently than cluster 1 and 69)
      Thus, we say that if i < params.n_clusters / 2, then i is a data bit, and if i >= params.n_clusters / 2, then i is a parity check bit
    """
    assert data.parity_check_matrix is not None, "Must have a parity check matrix"

    H_cluster: npt.NDArray = data.parity_check_matrix
    params = data.params

    # For simplicity, we will use an adjacency matrix for now. Later we can flatten this data-structure to make it cheaper
    if data.rounds is not None and len(data.rounds) > 0:
        primary_focuses_msgs_last = data.rounds[-1]
    else:
        data.rounds = []
        primary_focuses_msgs_last = [
            ["" for _ in range(params.n_clusters)] for _ in range(params.n_clusters)]

    async def pc_to_bit(i):
        assert i >= params.n_clusters / 2, "Must be a parity check bit"
        # H_ind = np.where(check_inds == i)[0][0]
        pc_ind = i - int(params.n_clusters / 2)
        neighbors = np.where(H_cluster[pc_ind, :] == 1)[0]
        # cluster_neighbor_inds = bit_inds[neighbors]
        p = ["" for _ in range(params.n_clusters)]

        for neighbor_ind in range(params.cluster_cluster_deg):
            neighbors_without_neighbor = np.delete(neighbors, neighbor_ind)

            ret = await local_neighbor_with_descr_labels(get_theorems_in_group(embeddings, data.cluster_labels, i, max_size=params.max_sample_size), primary_focuses_msgs_last[i][neighbor_ind],
                                                         [get_theorems_in_group(embeddings, data.cluster_labels, j, max_size=params.max_sample_size)
                                                          for j in neighbors_without_neighbor], [primary_focuses_msgs_last[j][i] for j in neighbors_without_neighbor], llm=llm)

            # primary_focuses_msgs[i][cluster_neighbor_inds[neighbor_ind]] = ret
            p[neighbors[neighbor_ind]] = ret
        return (i, p)

    async def bit_to_pc(i):
        assert i < params.n_clusters / 2, "Must be a data bit"
        # We offset by n_clusters / 2 because we want to start at the parity check bits
        offset = int(params.n_clusters / 2)
        neighbors = offset + np.where(H_cluster[:, i] == 1)[0]
        # print("Neighbors", neighbors, H_cluster.shape)
        p = ["" for _ in range(params.n_clusters)]

        for neighbor_ind in range(params.cluster_cluster_deg):
            neighbors_without_neighbor = np.delete(neighbors, neighbor_ind)

            ret = await local_neighbor_with_descr_labels(get_theorems_in_group(embeddings, data.cluster_labels, i, max_size=params.max_sample_size), primary_focuses_msgs_last[i][neighbor_ind],
                                                         [get_theorems_in_group(embeddings, data.cluster_labels, j, max_size=params.max_sample_size)
                                                          for j in neighbors_without_neighbor], [primary_focuses_msgs_last[j][i] for j in neighbors_without_neighbor], llm=llm)
            p[neighbors[neighbor_ind]] = ret
        return (i, p)

    for round_numb in range(data.completed_rounds, params.n_rounds):
        print(f"Starting BP Round {round_numb + 1} out of {params.n_rounds}")
        SKIP = 5
        tmp = []
        for i in range(0, params.n_clusters, SKIP):
            tasks = []
            for skip in range(min(SKIP, params.n_clusters - i)):
                # Then we have a bit
                if i + skip < params.n_clusters / 2:
                    tasks.append(bit_to_pc(i + skip))
                else:
                    tasks.append(pc_to_bit(i + skip))
                print("Appended cluster", i + skip)
            rets = await asyncio.gather(*tasks)
            rets_str = "\n\n".join(
                ["\n".join(list(filter(lambda x: x != "", r[1]))) for r in rets])
            print(
                f"\nReturns for BP round {round_numb + 1} out of {params.n_rounds} and cluster {i} to {i + SKIP - 1} (inclusive): {rets_str}\n")
            tmp = tmp + (rets)

        tmp.sort(key=lambda x: x[0])
        # sorted = np.array(tmp)[np.argsort(np.array([a[0] for a in tmp]))]
        primary_focuses_msgs = [a[1] for a in tmp]
        print(primary_focuses_msgs)

        data.rounds.append(primary_focuses_msgs)
        data.completed_rounds += 1

        save_dict(params, data)
        primary_focuses_msgs_last = copy.deepcopy(primary_focuses_msgs)
    return data


def run_hard_decision_function(llm: LLM, data: RunData, bp_round=-1):
    def shorten(cluster_idx: int, ind: int):
        # ind = 0
        # print(d_out["rounds"])
        last_msgs = filter(lambda a: a != "", data.rounds[ind][cluster_idx])
        joined = "\n".join(last_msgs)
        prompt = f"""Given the following descriptions of a primary set and the differences between a primary set of theorems and an adjacent set of theorems, what is the primary focus of the primary set of theorems in one or two sentence?

  DIFFERENCES: "{joined}"

  SHORTENED PRIMARY FOCUS:"""
        return llm(prompt, stop=STOP_DEFAULT_TOKENS)

    def shorten_all():
        if data.shortened is None:
            data.shortened = []
        # ind = -1
        ind = bp_round
        data.shortened.append([])
        data.shortened[-1] = ["BP Round " +
                              str((len(data.rounds) + ind) % len(data.rounds)), []]
        for i in range(data.params.n_clusters):
            data.shortened[-1][1].append(shorten(i, bp_round))
            print("Shortened", i, data.shortened[-1][1][-1])
        save_dict(data.params, data)

    shorten_all()

# We can check out **how much** having consecutive rounds matters via shortening different message rounds


async def run_bp_labeling(n_clusters: int, params: RunParams, thm_embs: custom_types.Embeddings, llm: LLM):
    """
      Runs BP on the given theorems, returning the labels for each theorem
    """
    assert n_clusters % 2 == 0, "Must have an even number of clusters"
    # Cluster with the number of dimensions equal to the number of embeddings
    _, labels, _unique_label_set = cluster.cluster(thm_embs, n_clusters)
    H = parity_check_matrix(
        int(n_clusters / 2), params.cluster_cluster_deg, params.cluster_cluster_deg)
    data = RunData(cluster_labels=labels, parity_check_matrix=H, params=params)
    await llm_bp(thm_embs, llm, data)

async def run_from_file(thm_embs: custom_types.Embeddings, file_path: str, llm: LLM, n_rounds=None):
    """
      Runs BP on the given theorems, returning the labels for each theorem
    """
    _data = json.load(open(file_path, "r"))
    data = RunData.from_dict(_data)
    # params = RunParams(**_data["params"])
    if n_rounds is not None:
        print("SETTING ROUNDS")
        data.params.n_rounds = n_rounds

    await llm_bp(thm_embs, llm, data)

if __name__ == "__main__":
    # await run_bp_labeling(16, thm_embs, llm)
    loop = asyncio.get_event_loop()
    file_path = f"data_store/embeddings_seed_69420_size_10000.json"
    embeddings: List[Tuple[str, List[float]]] = json.load(open(file_path, "r"))
    # thm_embs =
    # TODO: CHANGE
    n_clusters = 30
    bp_rounds = 3
    params = RunParams(n_clusters=n_clusters, seed=42_42_43, n_rounds=bp_rounds,
                       model_name=llm_name, max_sample_size=20, cluster_cluster_deg=3)
    if False:
        loop.run_until_complete(run_bp_labeling(n_clusters, params, embeddings, llm))
    elif True:
        new_n_rounds = 5
        loop.run_until_complete(run_from_file(
            embeddings, get_data_file_name(params), llm, n_rounds=new_n_rounds))
    elif False:
        _data = json.load(open(get_data_file_name(params), "r"))
        data = RunData.from_dict(_data)
        for i in range(5):
          run_hard_decision_function(llm, data, bp_round=i)
