import time
import os
import random
import json
from collections import deque
from copy import deepcopy
from itertools import chain

import requests
import networkx
import openai
import tqdm
import requests
import wikipedia
from nltk.corpus import wordnet as wn
import queue
import pprint
import networkx as nx
from queue import Queue
import torch
from rich.progress import track
import multiprocessing as mp

from utils import sample_neighbors, chatgpt_judge, partition_array


def mkdataset_tmn(path_meta, path_data, dir):
    d = torch.load(path_meta)
    whole, g, names, descriptions, train, eva, test = d['whole'], d['g'], d['names'], d['descriptions'], d['train'], d[
        'eva'], d['test']
    embeddings = torch.load(path_data)
    terms_dir = os.path.join(dir + '.terms')
    taxo_dir = os.path.join(dir + '.taxo')
    embed_dir = os.path.join(dir + '.terms.embed')
    train_dir = os.path.join(dir + '.terms.train')
    val_dir = os.path.join(dir + '.terms.validation')
    test_dir = os.path.join(dir + '.terms.test')

    with open(terms_dir, 'w') as f:
        for n in train + test + eva:
            f.write(str(n) + '\t' + names[n] + '\n')

    with open(taxo_dir, 'w') as f:
        for e in whole.edges():
            f.write('{}\t{}\n'.format(e[0], e[1]))

    with open(embed_dir, 'w') as f:
        f.write('{} {}\n'.format(len(names), embeddings[0].shape[-1]))
        for n in train + test + eva:
            line = str(n) + ' ' + ' '.join(map(lambda x: str(x), embeddings[n][0].numpy().tolist())) + '\n'
            f.write(line)

    with open(train_dir, 'w') as f:
        for n in train:
            f.write(str(n) + '\n')

    with open(val_dir, 'w') as f:
        for n in eva:
            f.write(str(n) + '\n')

    with open(test_dir, 'w') as f:
        for n in test:
            f.write(str(n) + '\n')


def extract_tree_from_imagenet(word_path, imagenet_path, save=True):
    tree = json.load(open(word_path))

    imagenet_labels = [l.lower().replace(' ', '_') for l in os.listdir(imagenet_path)]
    contained = []

    def dfs(head):
        flag = False
        if head['name'] in imagenet_labels and head['name'] not in contained:
            flag = True
            contained.append(head['name'])

        if len(head['children']) == 0:
            return flag
        else:
            lazy_del = []
            random.shuffle(head['children'])
            for c in head['children']:
                res = dfs(c)
                if not res:
                    lazy_del.append(c)
                flag = flag or res
            for c in lazy_del:
                head['children'].remove(c)
            return flag

    dfs(tree)
    print(len(contained))
    if save:
        with open(word_path.replace('wordnet', 'imagenet'), 'w') as f:
            json.dump(tree, f)

    return tree


def _get_holdout_subgraph(g, node_ids, node_to_remove=None):
    if node_to_remove is None:
        node_to_remove = [n for n in g.nodes if n not in node_ids]
    subgraph = g.subgraph(node_ids).copy()
    for node in node_to_remove:
        parents = set()
        children = set()
        ps = deque(g.predecessors(node))
        cs = deque(g.successors(node))
        while ps:
            p = ps.popleft()
            if p in subgraph:
                parents.add(p)
            else:
                ps += list(g.predecessors(p))
        while cs:
            c = cs.popleft()
            if c in subgraph:
                children.add(c)
            else:
                cs += list(g.successors(c))
        for p in parents:
            for c in children:
                subgraph.add_edge(p, c)
    # remove jump edges
    node2descendants = {n: set(nx.descendants(subgraph, n)) for n in subgraph.nodes}
    for node in subgraph.nodes():
        if subgraph.out_degree(node) > 1:
            successors1 = set(subgraph.successors(node))
            successors2 = set(chain.from_iterable([node2descendants[n] for n in successors1]))
            checkset = successors1.intersection(successors2)
            if checkset:
                for s in checkset:
                    if subgraph.in_degree(s) > 1:
                        subgraph.remove_edge(node, s)
    return subgraph


def remove_multiparents(graph):
    to_process = [n for n in graph.nodes if graph.in_degree(n) > 1]
    for n in to_process:
        edges = list(graph.in_edges(n))
        sel = random.choice(edges)
        edges.remove(sel)
        list(graph.remove_edge(p[0], p[1]) for p in edges)
    return graph


def mk_dataset_from_pickle(load_path, d_name, two_emb=False):
    import pickle
    import numpy as np
    with open(load_path, 'rb') as fin:
        data = pickle.load(fin)

    names = data['vocab']
    whole_g = data['g_full'].to_networkx()
    node_features = data['g_full'].ndata['x']
    train = data['train_node_ids']
    test = data['test_node_ids']
    val = data['validation_node_ids']
    if 'desc' in data.keys():
        desc = data['desc']
    roots = [node for node in whole_g.nodes() if whole_g.in_degree(node) == 0]
    # force root id to be 0
    if len(roots) > 1:
        root_vector = torch.mean(node_features[roots], dim=0, keepdim=True)
        whole_g = nx.relabel_nodes(whole_g, {i: i + 1 for i in whole_g.nodes})
        train = (np.array(train) + 1).tolist()
        test = (np.array(test) + 1).tolist()
        val = (np.array(val) + 1).tolist()
        roots = (np.array(roots) + 1).tolist()
        root = 0
        for r in roots:
            whole_g.add_edge(root, r)
        node_features = torch.cat((root_vector, node_features), 0)
        names = ['root'] + names
        train = [0] + train
    else:
        relabel = {i: i for i in whole_g.nodes}
        relabel[0] = roots[0]
        relabel[roots[0]] = 0
        assert 0 in train
        whole_g = nx.relabel_nodes(whole_g, relabel)

    whole_g = nx.DiGraph(whole_g)
    # whole_g = remove_multiparents(whole_g)
    tree = _get_holdout_subgraph(whole_g, train)

    # model, _ = clip.load('ViT-B/32', device='cuda')
    # model.eval()
    # descriptions = {}
    # for i, name in tqdm.tqdm(enumerate(names), desc='gen des'):
    #     sen = wn.synsets(name.replace(' ', '_'), pos=wn.NOUN)
    #     if len(sen) != 0:
    #         sen = sen[0].definition()
    #     else:
    #         try:
    #             sen = wikipedia.summary(name, sentences=1)
    #         except:
    #             sen = name
    #     descriptions[i] = sen

    # features = {}
    # for k, v in enumerate(tqdm.tqdm(names, desc='gen emb')):
    #     with torch.no_grad():
    #         features[k] = model.encode_text(clip.tokenize(v).cuda())
    features = {i: f.unsqueeze(0) for i, f in enumerate(node_features)}
    res = {'g': tree, 'whole': whole_g, 'train': train, 'test': test, 'eva': val,
           'names': names, 'descriptions': desc if 'desc' in data.keys() else None}
    if two_emb:
        plain_features = {i: f.unsqueeze(0) for i, f in enumerate(data['g_full'].ndata['y'])}
        torch.save(plain_features, d_name + '.pfeature.pt')

    torch.save(res, d_name + '.pt')
    torch.save(features, d_name + '.feature.pt')


def split_tree_dataset(whole_tree_path, split=0.8):
    tree = json.load(open(whole_tree_path))
    _id = [0]
    edge = []
    descriptions = []
    names = []

    def dfs(head):
        if len(head['children']) == 0:
            head['id'] = _id[0]
            _id[0] += 1
        else:
            head['id'] = _id[0]
            _id[0] += 1
            for i, child in enumerate(head['children']):
                dfs(child)
            list([edge.append([head['id'], c['id']]) for c in head['children']])

    def bfs(tree):
        tree['id'] = _id[0]
        _id[0] += 1
        q = Queue()
        q.put(tree)
        while not q.empty():
            head = q.get()
            names.append(head['name'])
            descriptions.append(head['definition'])

            for c in head['children']:
                c['id'] = _id[0]
                _id[0] += 1
                edge.append([head['id'], c['id']])
                q.put(c)

    def reconstruct():
        for node in test_eval:
            children = G.successors(node)
            father = list(G.predecessors(node))[0]
            G.remove_node(node)
            for c in children:
                G.add_edge(father, c)

    bfs(tree)

    G = nx.DiGraph()
    G.add_edges_from(edge)

    whole_g = deepcopy(G)

    start_id = count_n_level_id(G, 6)

    test_eval = random.sample(range(start_id, _id[0]), k=int(_id[0] * (1 - split)))
    train = list(range(_id[0]))
    list(map(lambda x: train.remove(x), test_eval))
    test, eva = test_eval[:len(test_eval) // 2], test_eval[len(test_eval) // 2:]

    reconstruct()

    return whole_g, G, names, descriptions, train, test, eva


def count_n_level_id(G, n):
    levels = [[0]]
    for i in range(n):
        fathers = levels[i]
        levels.append([])
        for f in fathers:
            levels[i + 1] += list(G.successors(f))

    res = []
    for l in levels:
        res += l
    return max(res) + 1


def mk_label_map():
    label_path = 'labels'
    label_map = {}
    with open(label_path, 'r') as f:
        res = f.readlines()
        for line in res:
            _id, label = line.split(':')
            label_map[_id] = [l.strip() for l in label.split(',')]

    return label_map


def mk_bamboo_map():
    search_api = 'https://opengvlab.shlab.org.cn/api/search'
    form_data = {'keyword': None}
    bamboo_path = 'cls/train/images'
    id_map_path = '../cls/id_map/id2name.json'
    bamboo_dataset = {}
    with open(id_map_path, 'r') as f:
        id_map = json.load(f)
    for l in tqdm.tqdm(os.listdir(bamboo_path)):
        if l in id_map:
            form_data['keyword'] = id_map[l][0]
            while True:
                try:
                    rep = requests.post(search_api, params=form_data)
                    if rep.status_code == 200:
                        if rep.json()['result']['matching']:
                            des = rep.json()['result']['matching'][0]['desc']
                            if des == '':
                                try:
                                    des = wikipedia.summary(id_map[l][0], sentences=1)
                                except:
                                    des = id_map[l][0]
                            bamboo_dataset[l] = {'name': id_map[l], 'descriptions': des,
                                                 'train': [os.path.join(bamboo_path, l, i) for i in
                                                           os.listdir(os.path.join(bamboo_path, l))]}
                    break
                except Exception as e:
                    print(e)
                    time.sleep(10)
    with open('bamboo_dataset.json', 'w') as f:
        json.dump(bamboo_dataset, f)


def build_dataset(label_map, sample_image=100):
    image_path = os.path.abspath('../imagenet')
    dataset = {}
    num_images = sample_image
    for k, v in tqdm.tqdm(label_map.items()):
        img_dir = os.path.join(image_path, k)
        sampled_imgs = random.choices(os.listdir(img_dir), k=3 * num_images)
        descriptions = [wn.synsets(vv.replace(' ', '_'), pos=wn.NOUN)[0].definition() for vv in v]
        dataset[k] = {'name': v, 'descriptions': descriptions,
                      'train': [os.path.join(img_dir, p) for p in sampled_imgs[:num_images]],
                      'val': [os.path.join(img_dir, p) for p in sampled_imgs[num_images:2 * num_images]],
                      'test': [os.path.join(img_dir, p) for p in sampled_imgs[2 * num_images:]]}
    with open('../datasets_json/imagenet_dataset.json', 'w') as f:
        json.dump(dataset, f)


def wordnet_dataset():
    word_queue = queue.Queue()
    pos = wn.NOUN
    word_tree = {}
    word_count = {}
    word_queue.put((word_tree, wn.synsets('entity', pos=pos)[0]))
    bar = tqdm.tqdm()

    while not word_queue.empty():
        cur_node, cur_query = word_queue.get()
        cur_node['name'] = cur_query.name().split('.')[0]
        cur_node['definition'] = cur_query.definition()
        cur_node['children'] = []
        word_count[cur_node['name']] = 1 if cur_node['name'] not in word_count else word_count[cur_node['name']] + 1

        for c in cur_query.hyponyms():
            cur_node['children'].append({})
            word_queue.put((cur_node['children'][-1], c))
        bar.update(1)

    with open('wordnet_dataset.json', 'w') as f:
        json.dump(word_tree, f)
    with open('wordnet_count.json', 'w') as f:
        json.dump(word_count, f)


# return a dict containing the name - name+desc dict
def construct_tree_to_dict(tree, unique=False):
    d = {}

    def dfs(head, unique):
        if unique:
            if head['name'] in d.keys():
                if head['definition'] != d[head['name']]['definition']:
                    head['name'] = head['name'] + '#' + head['definition']
                    d[head['name']] = head
                else:
                    pass
            else:
                d[head['name']] = head
        else:
            d[head['name']] = head

        if len(head['children']) == 0:
            return
        else:
            for c in head['children']:
                dfs(c, unique)

    dfs(tree, unique)
    return d


def _p(taxo, nodes, i):
    local_list = []
    for _n in track(nodes, description='Gen paths'):
        local_list.extend(list(networkx.all_simple_paths(taxo, 'n00001740', _n)))
    print('Checkpoint of paths.')
    json.dump(local_list, open(os.path.join('../data/', 'paths{}.json'.format(i)), 'w'))


def _f(taxo, bamboo, path, i):
    clean_d = {}
    for p in track(path):
        neighbors = sample_neighbors(taxo, p[-2], p[-1])
        p_name = [bamboo['id2name'][pp][0] for pp in p]
        neighbors = [bamboo['id2name'][pp][0] for pp in neighbors]
        clean_d[str(p)] = chatgpt_judge('path', p_name, neighbors)
        time.sleep(0.3)
    print('saving {}'.format(i))
    json.dump(clean_d, open(os.path.join('../data/clean2/', 'clean2_{}.json'.format(i)), 'w'))


def build_bamboo_taxo(bamboo_path='data/bamboo_V4.json', save_path='../data/'):
    bamboo = json.load(open(bamboo_path, 'r'))
    id2name = {}

    if os.path.exists(os.path.join(save_path, 'id2name.json')):
        id2name = json.load(open(os.path.join(save_path, 'id2name.json'), 'r'))
    else:
        # filter non-English class and classes without desc
        from langdetect import detect
        for i, n in track(bamboo['id2name'].items(),
                          description='Filtering non english class and those without desc. '):
            res = i in bamboo['id2desc'].keys() and bamboo['id2desc'][i] != ''
            for nn in n:
                res = res and detect(nn + ',' + bamboo['id2desc'][i]) == 'en'
            if res:
                id2name[i] = n

        ## mid save
        print('Checkpoint of id2name.')
        json.dump(id2name, open(os.path.join(save_path, 'id2name.json'), 'w'))

    # build bamboo taxonomy
    taxonomy = networkx.DiGraph()
    for i in track(bamboo['father2child'].keys(), description='Building taxonomy'):
        for child in bamboo['father2child'][i]:
            taxonomy.add_edge(i, child)

    # clean taxonomy
    core_taxonomy = _get_holdout_subgraph(taxonomy, id2name.keys())
    core_taxonomy = remove_root_nodes(core_taxonomy, 'n00001740')

    # mismount concept filter
    paths, processes = [], []
    if os.path.exists(os.path.join(save_path, 'paths.json')):
        paths = json.load(open(os.path.join(save_path, 'paths.json'), 'r'))
    else:
        all_nodes = list(core_taxonomy.nodes)
        all_nodes_p = partition_array(all_nodes, 30)

        for i, nodes in enumerate(all_nodes_p):
            processes.append(mp.Process(target=_p, args=(core_taxonomy, nodes, i)))
        for p in processes:
            p.start()
        for p in processes:
            p.join()

        # for n in track(core_taxonomy.nodes, description='Filter mismount concept'):
        #     paths.extend(list(networkx.all_simple_paths(core_taxonomy, 'n00001740', n)))

        # mid save
        # print('Checkpoint of paths.')
        # json.dump(paths, open(os.path.join(save_path, 'paths.json'), 'w'))
    if os.path.exists(os.path.join(save_path, 'clean2.json')):
        clean2 = json.load(open(os.path.join(save_path, 'clean2.json'), 'r'))
    else:
        paths = partition_array(paths, 40)
        threads = []
        for i, pa in enumerate(paths):
            threads.append(mp.Process(target=_f, args=(core_taxonomy, bamboo, pa, i)))

        for t in threads:
            t.start()

        for t in threads:
            t.join()

    name2id = {}
    for k, v in bamboo['id2name'].items():
        name2id[v[0]] = k

    for k, v in clean2.items():
        k = eval(k)
        p = name2id[k[0]]
        c = name2id[k[1]]
        if not v and core_taxonomy.has_edge(p, c):
            core_taxonomy.remove_edge(p, c)

    scc = list(networkx.weakly_connected_components(core_taxonomy))
    largest_scc = max(scc, key=len)
    core_taxonomy = core_taxonomy.subgraph(largest_scc).copy()
    core_taxonomy = remove_root_nodes(core_taxonomy, 'n00001740')

    with open(os.path.join(save_path, 'bamboo.taxo'), 'w') as f:
        for e in core_taxonomy.edges:
            f.write('{}\t{}\n'.format(e[0], e[1]))

    with open(os.path.join(save_path, 'bamboo.terms'), 'w') as f, open(os.path.join(save_path, 'bamboo.desc'),
                                                                       'w') as f2:
        for n in core_taxonomy.nodes:
            f.write('{}\t{}\n'.format(n, bamboo['id2name'][n][0]))
            f2.write('{}\t{}\n'.format(n, bamboo['id2desc'][n]))

    # visual concept filter
    # visual_concepts = [n for n in core_taxonomy.nodes if
    #                    chatgpt_judge('visual', [bamboo['id2name'][n], bamboo['id2desc'][n]])]
    # core_taxonomy = _get_holdout_subgraph(core_taxonomy, list(visual_concepts))
    # root = [n for n in core_taxonomy.nodes if core_taxonomy.in_degree(n) == 0]
    # assert len(root) == 1


def remove_root_nodes(G, root_node):
    # G is a networkx DiGraph object
    # root_node is the node to keep as the root
    # returns a new DiGraph object with only one root node
    # assumes that G has at least one root node
    root_nodes = [n for n in G.nodes if G.in_degree(n) == 0]
    nodes_to_remove = []
    for r in root_nodes:
        if r != root_node:
            descendants = list(nx.descendants(G, r))
            nodes_to_remove.append(r)
            nodes_to_remove.extend(descendants)
    G.remove_nodes_from(nodes_to_remove)
    return G


def interactive_json_maker():
    name_to_pointer = {}
    tree = {}

    def load():
        if os.path.exists('../datasets_json/middle_handcrafted.json'):
            t = json.load(open('../datasets_json/middle_handcrafted.json'))
            n = construct_tree_to_dict(t)
            return t, n
        else:
            return {}, {}

    def insert_node(father, name, description):
        if father == '':
            assert len(tree.keys()) == 0
            tree['name'] = name
            tree['description'] = description
            tree['children'] = []
            name_to_pointer[name] = tree
        else:
            assert father in name_to_pointer.keys()
            name_to_pointer[father]['children'].append({
                'name': name,
                'description': description,
                'children': []
            })
            name_to_pointer[name] = name_to_pointer[father]['children'][-1]

    def save():
        with open('../datasets_json/middle_handcrafted.json', 'w') as f:
            json.dump(tree, f)

    tree, name_to_pointer = load()

    while True:
        command = input('whats your command? ')
        if command == 'q' or command == 'quit':
            break
        elif command == 'i' or command == 'insert':
            father = input('father: ').strip().lower()
            name = input('name: ').strip().lower()
            description = input('description: ').strip().lower()
            insert_node(father, name, description)
            print('insertion finished')
        elif command == 'p' or command == 'print':
            which = input('print which node? ').strip().lower()
            if which == '':
                pprint.pprint(tree)
            else:
                assert which in name_to_pointer.keys()
                pprint.pprint(name_to_pointer[which])
        elif command == 's' or command == 'save':
            save()
            print(len(name_to_pointer.keys()))
        elif command == 'l' or command == 'load':
            tree, name_to_pointer = load()
        elif command == 'd' or command == 'delete':
            which = input('print which node? ').strip().lower()
            assert which in name_to_pointer.keys()
            del name_to_pointer[which]
        elif command == 'n' or command == 'number':
            print(len(name_to_pointer.keys()))
        elif command == 'ls' or command == 'list':
            which = input('list which node children? ').strip().lower()
            assert which in name_to_pointer.keys()
            pprint.pprint([c['name'] for c in name_to_pointer[which]['children']])
        else:
            print('invalid')
            continue

    save()


def fast_embed(data_path):
    import fasttext
    import fasttext.util
    # fasttext.util.download_model('en', if_exists='ignore')  # English
    ft = fasttext.load_model('/data/home10b/xw/visualCon/QEN-main/crawl-300d-2M-subword.bin')
    with open(data_path, 'r') as f:
        lines = f.readlines()
    words = list(map(lambda x: x.split('\t')[-1].split('@')[0], lines))
    ids = list(map(lambda x: x.split('\t')[0], lines))
    embeds = list(map(lambda x: ft[x], words))
    res = str(len(embeds)) + ' 300\n'
    with open(data_path.replace('terms', 'embed'), 'w') as f:
        f.writelines(res)
        for i, e in tqdm.tqdm(zip(ids, embeds)):
            assert e.sum() != 0
            res = str(i) + ' ' + ' '.join(map(str, e.tolist())) + '\n'
            f.writelines(res)


def extract_conceptnet(path):
    with open(path + '/train100k.txt', 'r') as f:
        train = f.readlines()
    with open(path + '/test.txt', 'r') as f:
        test = f.readlines()
    with open(path + '/dev1.txt', 'r') as f:
        eva = f.readlines()
    isa = list(filter(lambda x: x.split('\t')[0].lower() == 'isa', train))
    test = list(filter(lambda x: x.split('\t')[0].lower() == 'isa', test))
    eva = list(filter(lambda x: x.split('\t')[0].lower() == 'isa', eva))
    g = nx.DiGraph()
    id2name = {}
    name2id = {}
    i = 1
    for relation in isa + test + eva:
        child, parent = relation.split('\t')[1], relation.split('\t')[2]
        if child not in name2id.keys():
            name2id[child] = i
            id2name[i] = child
            i += 1
        if parent not in name2id.keys():
            name2id[parent] = i
            id2name[i] = parent
            i += 1

        g.add_edge(name2id[parent], name2id[child])

    self_loops = nx.nodes_with_selfloops(g)
    for n in self_loops:
        g.remove_edge(n, n)

    cycle = nx.find_cycle(g)
    while len(cycle) > 0:
        d, s = cycle[0][0], cycle[-1][0]
        g.remove_edge(s, d)
        try:
            cycle = nx.find_cycle(g)
        except:
            break

    roots = [n for n in g.nodes() if g.in_degree(n) == 0]
    name2id['#root'] = 0
    id2name[0] = '#root'
    for r in roots:
        g.add_edge(0, r)
    leaves = [n for n in g.nodes() if g.out_degree(n) == 0]
    paths = list(filter(lambda x: len(x) > 2, [nx.shortest_path(g, 0, l) for l in leaves]))
    candidates = list(set(chain.from_iterable(list(map(lambda x: x[2:], paths)))))
    sample_size = min(120, int(0.01 * len(candidates)))
    random.seed(47)
    random.shuffle(candidates)
    test = candidates[:sample_size]
    eva = candidates[sample_size:2 * sample_size]
    train = list([n for n in g.nodes() if n not in test + eva])
    taxo = _get_holdout_subgraph(g, train)
    torch.save(
        {'train': train, 'test': test, 'eva': eva, 'g': taxo, 'whole': g, 'names': id2name, 'descriptions': None},
        'conceptnet.pt')


def mk_wn(path):
    random.seed(47)
    with open(path + 'wn.terms') as f:
        terms = f.readlines()
        terms = {int(term.split('\t')[1]): term.split('\t')[0] for term in terms}
        terms = [terms[i] for i in range(0, len(terms))]
    with open(path + 'wn.desc') as f:
        descs = f.readlines()
        descs = {int(desc.split('\t')[0]): desc.split('\t')[1] for desc in descs}
        descs = [descs[i] for i in range(0, len(descs))]
    with open(path + 'wn.taxo') as f:
        relations = f.readlines()

    g = nx.DiGraph()
    for r in relations:
        child, parent = r.split('\t')[0], r.split('\t')[1]
        g.add_edge(int(parent), int(child))

    terms = ['##wordnet_root##'] + terms
    descs = ['##wordnet_root##'] + descs
    g = nx.relabel_nodes(g, {i: i + 1 for i in g.nodes})
    roots = [n for n in g.nodes() if g.in_degree(n) == 0]

    for r in roots:
        g.add_edge(0, r)
    sampled = list(g.nodes())
    sampled.remove(0)
    random.shuffle(sampled)
    sz = min(1000, int(0.01 * len(terms)))
    eva = sampled[:sz]
    test = sampled[sz:2 * sz]
    train = [0] + sampled[2 * sz:]
    core_taxo = _get_holdout_subgraph(g, train, eva + test)
    torch.save(
        {'train': train, 'test': test, 'eva': eva, 'names': terms, 'descriptions': descs, 'g': core_taxo, 'whole': g},
        'wordnet.pt')

word_count = None
if __name__ == '__main__':
    # word_count = json.load(open('wordnet_count.json'))
    # word_tree = json.load(open('wordnet_dataset.json'))
    # pruned = word_tree_pruner(word_tree)
    # interactive_json_maker()
    # t = extract_tree_from_imagenet('/data/home10b/xw/visualCon/datasets_json/wordnet_dataset.json',
    #                                '/data/home10b/xw/imagenet21k/imagenet_images')
    # dt = construct_tree_to_dict(t, unique=True)
    # imagenet_labels = [l.lower().replace(' ', '_') for l in os.listdir('/data/home10b/xw/imagenet21k/imagenet_images')]

    # con = [k for k, _ in dt.items() if k in imagenet_labels]
    # print(len(con), con)
    # print(len(dt))
    # whole_g, G, names, descriptions, train, test, eva = split_tree_dataset(
    #     '/data/home10b/xw/visualCon/datasets_json/imagenet_dataset.json')
    # print(len(train), len(test), len(eva), len(names))
    #
    # torch.save({'whole': whole_g,
    #             'g': G, 'names': names, 'descriptions': descriptions, 'train': train, 'test': test, 'eva': eva
    #             }, 'imagenet_dataset.pt')
    # from datasets_torch.treeset import TreeSet
    # t = TreeSet(G, names, descriptions)
    # mkdataset_tmn('/data/home10b/xw/visualCon/imagenet_dataset.pt', '/data/home10b/xw/visualCon/tree_data.pt',
    #               '/data/home10b/xw/visualCon/TMN-main/data/mywn')
    # mk_dataset_from_pickle('/data/home10b/xw/visualCon/TMN-main/data/MAG-PSY/psychology.pickle.bin',
    #                        '../mag_psy', False)
    # mkdataset_tmn('/data/home10b/xw/visualCon/wn_verb.pt', '/data/home10b/xw/visualCon/wn_verb.feature.pt',
    #               '/data/home10b/xw/visualCon/TMN-main/data/SemEval-V/semeval_food.pickle.bin')
    # fast_embed('/data/home10b/xw/visualCon/TMN-main/data/mesh/mesh.terms')
    # build_bamboo_taxo('/data/home10b/xw/visualCon/data/bamboo_V4.json')
    mk_wn('/data/home10b/xw/visualCon/data/wn_full/')