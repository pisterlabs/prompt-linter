#!/usr/bin/env python3
import networkx as nx
import sqlite3
import logging
import json
import argparse
import pandas as pd
from pathlib import Path
from cdlib.algorithms import leiden
import umap
import numpy as np
import subprocess
import openai
from collections import Counter
import string

"""
Loads the org-roam database from the given path, and selects the file, title, and id from the nodes table, and the source and dest from the links table.

Output format is a tuple of (titles, links), where titles is a dictionary of file -> title, and links is a list of tuples of (source, dest)
"""
def load_from_db(path):
    logging.info(f"Loading from {path}")
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute('SELECT file, title, id FROM nodes')
    titles = c.fetchall()

    c.execute('SELECT source, dest FROM links')
    links = c.fetchall()

    c.execute('SELECT file, title FROM files')
    files = c.fetchall()

    master_titles = {}
    for filename, title in files:
        master_titles[filename] = title

    t = {}
    id_to_file = {}
    for file, title, or_id in titles:
        if 'private' in file:
            continue
        title = master_titles[file][1:-1]
        t[file] = title
        id_to_file[or_id] = file

    final_links = []
    for source, dest in links:
        if id_to_file.get(source, None) is None or id_to_file.get(dest, None) is None:
            continue
        final_links.append((id_to_file[source], id_to_file[dest]))
    return t, final_links

def generate_url(file_name, replace_dict):
    new_url = file_name
    for key, val in replace_dict.items():
        if val == "NONE":
            val = ""
        new_url = new_url.replace(key, val)
    return new_url.replace('"', '')

def parse_links(links, titles, top=None, replace_dict={}):
    logging.info(f"Parsing links")
    l = []
    for file1, file2  in links:
        if 'private' in file2 or 'private' in file2:
            continue
        file1_name = titles.get(file1)
        file2_name = titles.get(file2)
        if file1_name and file2_name:
            l.append({
                "source": file1_name,
                "source_url": generate_url(file1, replace_dict),
                "target": file2_name,
                "target_url": generate_url(file2, replace_dict),
            })
    #if top:
        #df = pd.DataFrame(l)
    return l

def color_nodes(community_dictionary, titles, links, replace_dict={}):
    # community_dictionary is a mapping of 'title' -> 'community'
    nodes = {}

    titles = {generate_url(k[1:-1], replace_dict): v for k, v in titles.items()}
    for link in links:
        source = link['source']
        target = link['target']
        if source not in nodes:
            url = link['source_url']
            nodes[source] = {
                'id': titles[url],
                'url': url,
                'group': community_dictionary[source]
            }
        if target not in nodes:
            url = link['target_url']
            nodes[target] = {
                'id': titles[url],
                'url': url,
                'group': community_dictionary[target]
            }
    return nodes

def generate_community_colors(titles, links, replace_dict={}, community_algo=leiden):
    logging.info(f"Generating community colors with algorithm {community_algo}")

    G = nx.Graph()
    for link in links:
        source = link['source']
        target = link['target']
        if source not in G:
            G.add_node(source)
            if target not in G:
                G.add_node(target)

        G.add_edge(source, target)

    community_sets = {}
    community_list = community_algo(G).communities
    for i, com in enumerate(community_list):
        for note_name in com:
            community_sets[note_name] = i

    nodes = color_nodes(community_sets, titles, links, replace_dict)
    return nodes, G

def dump(nodes, links, groups, name):
    logging.info(f"Writing json to {name}")
    output = {}

    for cur_link in links:
        cur_link["x1"] = nodes[cur_link["source"]]['x']
        cur_link["y1"] = nodes[cur_link["source"]]['y']
        cur_link["x2"] = nodes[cur_link["target"]]['x']
        cur_link["y2"] = nodes[cur_link["target"]]['y']


    output['links'] = links
    output['nodes'] = list(nodes.values())
    output["groups"] = groups

    with open(f"{name}.json", 'w') as f:
        json.dump(output, f)

def run_umap(nodes, links, name="org-data"):
    logging.info("Running dumping into node2vec format")
    node2vec_edgelist = []

    ids = range(1, len(nodes) + 1)

    node_to_id = dict(zip(nodes.keys(), ids))
    id_to_node = dict(zip(ids, nodes.keys()))
    for cur_link in links:
        edge = f'{node_to_id[cur_link["source"]]} {node_to_id[cur_link["target"]]}'
        node2vec_edgelist.append(edge)

    with open(f"{name}.edgelist", 'w') as f:
        f.write("\n".join(node2vec_edgelist))

    logging.info(f"Running node2vec on {name}")
    #node2vec -i:/workspace/org-data.edgelist  -o:/workspace/org-data.emb  -d:64 -l:40 -q:0.5
    subprocess.run(["node2vec", f'-i:/workspace/{name}.edgelist', f"-o:/workspace/{name}.emb", "-d:64", "-l:40", "-q:0.5",])

    logging.info(f"Running UMAP on {name}")
    f = f"{name}.emb"
    data = np.genfromtxt(f, delimiter=" ", skip_header=1)
    mapper = umap.UMAP(spread=3.0, min_dist=0.5, n_neighbors=100)
    u = mapper.fit_transform(data[:,1:])

    x = u[:, 0]
    y = u[:, 1]

    # determine the minimum and maximum for x and y separately
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)

    # normalize x and y separately
    x_normalized = (x - min_x) / (max_x - min_x)
    y_normalized = (y - min_y) / (max_y - min_y)

    # Recombine the normalized x and y into a single array
    data_normalized = np.column_stack((x_normalized, y_normalized))

    id_to_position = dict(zip(data[:, 0], data_normalized))

    for key, value in id_to_position.items():
        node_name = id_to_node[int(key)]
        nodes[node_name]['num_id'] = int(key)
        nodes[node_name]['x'] = str(value[0])
        nodes[node_name]['y'] = str(value[1])

    return nodes


def generate_positions(G, nodes, iterations=50):
    logging.info(f"Generating and iterating through spring layout with iterations {iterations}")
    pos = nx.spring_layout(G, scale=2, k=0.1, iterations=iterations)
    for key, value in pos.items():
        if key in nodes:
            nodes[key]["x"] = value[0]
            nodes[key]["y"] = value[1]
    return nodes

def generate_group_names(nodes, links):
    logging.info("Generating group names")
    df = pd.DataFrame(nodes.values())
    links_df = pd.DataFrame(links)
    links_df["count"] = 1
    name_to_edges_count = Counter(links_df.groupby("source").sum()["count"].to_dict())
    name_to_edges_count += Counter((links_df.groupby("target").sum()["count"].to_dict()))
    df["edge_counts"] = df["id"].apply(lambda x: name_to_edges_count.get(x, 0))  

    def generate_prompt(df, group):
        df_sorted = df.sort_values("edge_counts", ascending=False)
        df_sorted = df_sorted[df_sorted["group"] == group]
        central_node = df_sorted["id"].iloc[0]
        group_names = df_sorted["id"][:64].to_list()
        prompt_start = "for the following list, generate a one to two word name for the entire category, that's clear without any punctuation, that I can use as a map legend: ["
        prompt_start += ",".join(group_names)
        prompt_start += "]"
        return prompt_start, central_node

    res = {}
    for group in df["group"].unique():
        test_prompt, central_node = generate_prompt(df, group)
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=test_prompt
        )
        response.choices[0]["text"]
        res[str(group)] = {
            "name": response.choices[0]["text"].strip().translate(str.maketrans('', '', string.punctuation)),
            "central_node": central_node
        }

    logging.info(f"Generated group names {res}")
    return res

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Generates a json file from your org-roam DB")
    parser.add_argument("--org-db-location",  help="Location of org-roam.db file. Defaults to $HOME/.emacs.d/org-roam.db", type=str, default=f"{Path.home()}/.emacs.d/org-roam.db", dest="db_location")
    parser.add_argument("--output", "-o", help="File to output as. Defaults to './org-data.json'", type=str, default="./org-data", dest="output_location")
    parser.add_argument("--replace", dest="replacements", nargs="+", help="Replacement to generate urls. Takes in <FILE_PATH> <REPLACEMENT_VALUE>")
    parser.add_argument("--top", default=None, dest="top", help="Number of nodes to cut off by. Default is to generate all nodes")
    parser.add_argument("--generate-groups", default=False, action="store_true", dest="generate_groups", help="Generate groups based on file name. Uses the titles of the top 64 nodes with the most edges to generate a prompt for OpenAI to generate a name for the group. `OPENAI_API_KEY` must be set as an environment variable.")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("debug.log"),
            logging.StreamHandler()
        ]
    )
    print(args)

    if args.replacements and len(args.replacements) % 2 != 0:
        print("Replacements must be in pairs")
        exit(1)

    logging.info(f"Loading db from {args.db_location}")
    titles, links = load_from_db(path=args.db_location)
    if args.replacements:
        replacements = {args.replacements[i]: args.replacements[i+1] for i in range(0, len(args.replacements), 2)}
    else:
        replacements = {}
    logging.info(f"Replacing according to {replacements}")
    links = parse_links(links, titles, args.top, replacements)
    nodes, G = generate_community_colors(titles, links, replacements)
    nodes = generate_positions(G, nodes, iterations=200)

    group_names = {}
    if args.generate_groups:
        group_names = generate_group_names(nodes, links)

    dump(nodes, links, group_names, name=args.output_location)
