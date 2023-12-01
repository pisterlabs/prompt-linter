import os
import re
import numpy as np
from openai.embeddings_utils import cosine_similarity, get_embedding as _get_embedding
from tenacity import  stop_after_attempt, wait_random_exponential
import openai
import jraph
import pickle
import jax
import jax.numpy as jnp
from typing import List, Tuple
import time 



openai.api_key = OPENAI_API_KEY

get_embedding = _get_embedding.retry_with(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))


FEATURE_NUM = 1536
EDGE_FEATURE_NUM = 4

# Return a list of dictionaries describing adjacency relationships
def get_adj_dict(vuln: str) -> List[dict]:
    inputFileDir = f"./binary_graph_data/{vuln}/edge/"
    dirs = os.listdir(inputFileDir)
    dirs.sort(key=lambda x: int(x[:-4]))

    edge_type = {"red": 0, "green": 1, "blue": 2, "cyan": 3}
    adj_dict_list = []
    for file in dirs:
        inputFilePath = inputFileDir + file
        f = open(inputFilePath, "r")
        lines = f.readlines()

        adj_dict = {}
        edge_id = 0
        edge_id_dict = {}
        for line in lines:
            if "block" in line:
                # Split the line into block_id and block_dest_id
                block_id, block_dest_id = line.split(" -> ")

                # Cleaning to get block_id, block_dest_id, edge_label
                block_id = block_id.strip()
                block_dest_id, edge_label = block_dest_id.split("[color=")
                block_dest_id = block_dest_id.strip()
                edge_label = edge_label.strip()[:-1]


                if block_id not in adj_dict:
                    adj_dict[block_id] = []
                adj_dict[block_id].append((block_dest_id, edge_type[edge_label]))
        # for k,v in adj_dict.items():
        #     print(f"{k}:{v}")
        adj_dict_list.append(adj_dict)
    return adj_dict_list        

# Return a list of dict of mapping block_id to its corresponding bytecode
def get_x_dict(vuln: str) -> List[dict]:
    inputFileDir = f"./binary_graph_data/{vuln}/new_node/"
    dirs = os.listdir(inputFileDir)
    dirs.sort(key=lambda x: int(x[:-4]))

    x_dict_list = []

    for file in dirs:
        inputFilePath = inputFileDir + file
        f = open(inputFilePath, "r")
        lines = f.readlines()

        ### Getting X
        X_dict = {}

        for line in lines:
            if "block" in line:
                block_id, label = line.split(" [label=")
                block_id = block_id.strip()
                label = label.strip().strip('"')
                label = label[:-2].strip()
                X_dict[block_id] = label
        # for k,v in X_dict.items():
        #     print(f"{k}:{v}")
        x_dict_list.append(X_dict)
    return x_dict_list

def encoder(code: str) -> np.ndarray:
    text = "This is a block of EVM bytecode: " + code
    model_id = "text-embedding-ada-002"
    emb = openai.Embedding.create(input=[text], model=model_id)['data'][0]['embedding']
    emb_np = np.array(emb)
    return emb_np
    # return np.ones((FEATURE_NUM,))

def make_jraph_dataset(vuln):
    x_dict_list = get_x_dict(vuln)
    adj_dict_list = get_adj_dict(vuln)
    print(len(x_dict_list), len(adj_dict_list))
    graph_list = list(zip(x_dict_list, adj_dict_list))
    print(len(graph_list))
    dataset = []
    print("Total # of graphs", len(graph_list))
    for index, (x_dict, adj_dict) in enumerate(graph_list):
        print(f"Processing graph {index}")
        # Sanity check
        for key in adj_dict.keys():
            if key not in x_dict.keys():
                raise Exception(f"Key {key} not in x_dict")
        
        # Define final dataset
        n = len(x_dict)

        blk_id_to_index = {blk_id: i for i, (blk_id, _) in enumerate(x_dict.items())}
        nodes = []
        edges = []
        senders = []
        receivers = []

        try:
            # Convert dicts to jraph representation
            for i, (blk,code) in enumerate(x_dict.items()):
                nodes.append(encoder(code))
                if blk in adj_dict:
                    for (dest, edge_type) in adj_dict[blk]:
                        edge_one_hot = np.zeros((EDGE_FEATURE_NUM,))
                        edge_one_hot[edge_type] = 1
                        edges.append(edge_one_hot)
                        senders.append(blk_id_to_index[blk])
                        receivers.append(blk_id_to_index[dest])
        except Exception as e:
            print(f"Error in graph {index}: {e}")
            with open(f"./dataset/{vuln}_{index}.pkl", "wb") as f:
                pickle.dump(dataset, f)
            continue

        # Convert to jraph
        graph = jraph.GraphsTuple(
                                  n_node=np.array([len(nodes)]),
                                  n_edge=np.array([len(edges)]),
                                  nodes=np.array(nodes), 
                                  edges=np.array(edges), 
                                  senders=np.array(senders), 
                                  receivers=np.array(receivers), 
                                  globals=np.array([1]),
                                )
        target = [1]
        dataset.append({"input_graph": graph, "target": target})
    with open(f"./dataset/{vuln}.pkl", "wb") as f:
        pickle.dump(dataset, f)


def re_pack_dataset(vuln, numbers):
    x_dict_list = get_x_dict(vuln)
    adj_dict_list = get_adj_dict(vuln)
    print(len(x_dict_list), len(adj_dict_list))
    graph_list = list(zip(x_dict_list, adj_dict_list))
    print(len(graph_list))
    dataset = []
    print("Total # of graphs", len(graph_list))
    for index in numbers:
        (x_dict, adj_dict) = graph_list[index]
        print(f"Processing graph {index}")

        # Sanity check
        for key in adj_dict.keys():
            if key not in x_dict.keys():
                raise Exception(f"Key {key} not in x_dict")
        
        blk_id_to_index = {blk_id: i for i, (blk_id, _) in enumerate(x_dict.items())}
        nodes = []
        edges = []
        senders = []
        receivers = []

        # Convert dicts to jraph representation
        for i, (blk,code) in enumerate(x_dict.items()):
            nodes.append(encoder(code))
            if blk in adj_dict:
                for (dest, edge_type) in adj_dict[blk]:
                    edge_one_hot = np.zeros((EDGE_FEATURE_NUM,))
                    edge_one_hot[edge_type] = 1
                    edges.append(edge_one_hot)
                    senders.append(blk_id_to_index[blk])
                    receivers.append(blk_id_to_index[dest])
        
        # Convert to jraph
        graph = jraph.GraphsTuple(
                                  n_node=np.array([len(nodes)]),
                                  n_edge=np.array([len(edges)]),
                                  nodes=np.array(nodes), 
                                  edges=np.array(edges), 
                                  senders=np.array(senders), 
                                  receivers=np.array(receivers), 
                                  globals=np.array([1]),
                                )
        target = [1]
        dataset.append({"input_graph": graph, "target": target})
    print(f"Length of dataset for {vuln}: ", len(dataset))
    with open(f"./dataset/{vuln}_fixed.pkl", "wb") as f:
        pickle.dump(dataset, f)


def get_numpy_dataset():
    export_dir = "./dataset/"

    x_dict_list = get_x_dict("")
    adj_dict_list = get_adj_dict("")
    print(len(x_dict_list), len(adj_dict_list))
    graph_list = list(zip(x_dict_list, adj_dict_list))
    print(len(graph_list))
    numpy_graphs = []

    for index, (x_dict, adj_dict) in enumerate(graph_list):

        # Sanity check
        for key in adj_dict.keys():
            if key not in x_dict.keys():
                raise Exception(f"Key {key} not in x_dict")

        # Define final dataset
        n = len(x_dict)
        x = np.zeros((n, FEATURE_NUM))
        adj = np.zeros((n,n))
        edge_feat = np.zeros((n,n, EDGE_FEATURE_NUM))

        blk_id_to_index = {blk_id: i for i, (blk_id, _) in enumerate(x_dict.items())}
        
        # Convert dicts to numpy array representation
        for i, (blk,code) in enumerate(x_dict.items()):
            x[i,:] = encoder(code)
            # print(code)
            if blk in adj_dict:
                for (dest, edge_type) in adj_dict[blk]:
                    adj[i, blk_id_to_index[dest]] = 1
                    edge_feat[i, blk_id_to_index[dest], edge_type] = 1

        file_path = os.path.join(f"{export_dir}/features", f"X_{index}.npy")
        np.save(file_path, x)
        file_path = os.path.join(f"{export_dir}/adj", f"adj_{index}.npy")
        np.save(file_path, adj)
        file_path = os.path.join(f"{export_dir}/edge_features", f"edge_feat_{index}.npy")
        np.save(file_path, edge_feat)

        numpy_graphs.append((x, adj, edge_feat))
    return numpy_graphs


if __name__=="__main__":
    # get_adj_dict(0)
    vuln_list = ["delegatecall", "integeroverflow", "reentrancy", "timestamp"]
    # for vuln in vuln_list:
    #     make_jraph_dataset(vuln)

    vuln = "timestamp"
    re_pack_dataset(vuln,[109, 123])

    model_id = "text-embedding-ada-002"

    

    with open(f"./dataset/{vuln}.pkl", "rb") as f:
        dataset = pickle.load(f)
    
    with open(f"./dataset/{vuln}_fixed.pkl", "rb") as f:
        additional = pickle.load(f)

    for g in additional:
        dataset.append(g)

    with open(f"./dataset/{vuln}_fixed.pkl", "wb") as f:
        pickle.dump(dataset, f)

    print(len(dataset))

    # compute the embedding of the text
    # text = "This is a block of EVM bytecode:  d: PUSH1 0x0 f: CALLDATALOAD  10: PUSH29 0x100000000000000000000000000000000000000000000000000000000 2e: SWAP1  2f: DIV  30: PUSH4 0xffffffff 35: AND  36: DUP1  37: PUSH4 0x19165587 3c: EQ  3d: PUSH2 0x5c 40: JUMPI"
    # emb = openai.Embedding.create(input=[text], model=model_id)['data'][0]['embedding']
    # print(emb)
    # emb_np = np.array(emb)
    # print(emb_np)
    # print(emb_np.shape)
    # print(emb_np + np.zeros((FEATURE_NUM,)))
    # print(len(emb))

