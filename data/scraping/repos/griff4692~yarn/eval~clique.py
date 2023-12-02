from collections import defaultdict, Counter
import os
from datetime import datetime
import ujson
import itertools
import pickle
import sys
from sklearn.metrics.pairwise import cosine_similarity
import regex as re

from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from datasets import load_from_disk
import argparse
import numpy as np
import pandas as pd
np.random.seed(1992)
import json
from nltk import word_tokenize
import spacy
import torch
from evaluate import load
import stanza
from nltk.corpus import stopwords
import string
from eval.model_loader import add_args, load_model_and_apply_patches
from transformers import pipeline

SAP_BERT = 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'
META_COLS = ['plan_recall', 'plan_precision', 'plan_f1']
IN_DIR = '/nlp/projects/summarization/bhc_data_cleanup'
SPAN_EMBED_DIM = 768
_DEFAULT_PRED_ENT_THRESHOLD = 0.75
ENT_MERGE_THRESHOLD = 0.75
BHC_PREFIX = '\n\n### PARTIAL HOSPITAL COURSE:\n'
BHC_FULL = '\n\n### BRIEF HOSPITAL COURSE:\n'


PATIENT_TERMS = {'patient', 'pt', 'patient\'s', 'patients', 'patients\''}
BHC_STOPWORDS = set(stopwords.words('english')).union(string.punctuation).union(PATIENT_TERMS)


def remove_dup_lines(text):
    arr = text.split('\n')
    new_arr = []
    for x in arr:
        x = x.strip()
        if x not in new_arr and len(x) > 0:
            new_arr.append(x)
    return '\n'.join(new_arr)


def gen_from_clique(pipe, prompt):
    n = len(prompt)
    response = pipe(
        prompt, num_return_sequences=1, max_new_tokens=args.max_new_tokens,
    )[0]["generated_text"][n:]

    response = remove_dup_lines(response)
    return response


# https://www.geeksforgeeks.org/connected-components-in-an-undirected-graph/
class Graph:
    # init function to declare class variables
    def __init__(self, V):
        self.V = V
        self.adj = [[] for _ in range(V)]

    def DFSUtil(self, temp, v, visited):

        # Mark the current vertex as visited
        visited[v] = True

        # Store the vertex to list
        temp.append(v)

        # Repeat for all vertices adjacent
        # to this vertex v
        for i in self.adj[v]:
            if visited[i] == False:
                # Update the list
                temp = self.DFSUtil(temp, i, visited)
        return temp

    # method to add an undirected edge
    def addEdge(self, v, w):
        self.adj[v].append(w)
        self.adj[w].append(v)

    # Method to retrieve connected components
    # in an undirected graph
    def connectedComponents(self):
        visited = []
        cc = []
        for i in range(self.V):
            visited.append(False)
        for v in range(self.V):
            if visited[v] == False:
                temp = []
                cc.append(self.DFSUtil(temp, v, visited))
        return cc


def remove_non_ents(text, remove_title=True):
    lines = text.split('\n')
    new_lines = []
    for line in lines:
        if 'title:' in line.lower() and not remove_title:
            new_lines.append(line)
        elif '<doc-sep>' in line.lower() and not remove_title:
            new_lines.append(line)
        elif line == '':
            new_lines.append(line)
        elif '{{' in line or '<e>' in line:
            new_lines.append(line)
    return '\n'.join(new_lines)


def remove_non_ents_from_list(text, rel_spans, remove_title=True):
    rel_spans_lower = [x.strip(string.punctuation).lower() for x in list(rel_spans)]
    lines = text.split('\n')
    new_lines = []
    for line in lines:
        if 'title:' in line.lower() and not remove_title:
            new_lines.append(line)
        elif '<doc-sep>' in line.lower() and not remove_title:
            new_lines.append(line)
        elif line == '':
            new_lines.append(line)
        elif any([x_lower in text.lower() for x_lower in rel_spans_lower]):
            new_lines.append(line)
    return '\n'.join(new_lines)


def sents_from_html(html_str):
    tps = html_str.split('<SEP>')
    return [tps[idx + 1] for idx, tp in enumerate(tps) if tp.startswith('<s') and idx + 1 < len(tps)]


def split_into_notes(html_str):
    tps = html_str.split('<SEP>')
    notes = []
    curr_note = []
    for tp in tps:
        curr_note.append(tp)
        if tp == '</d>':
            notes.append('<SEP>'.join(curr_note))
            curr_note = []
    return notes


def take_latest(source_transform, max_toks=8192):
    curr_num_toks = 0
    lines = source_transform.split('\n')
    trunc = []
    for end_idx in range(len(lines) - 1, -1, -1):
        line = lines[end_idx]
        curr_num_toks += len(line.split(' '))
        trunc.insert(0, line)
        if curr_num_toks > max_toks:
            break
    source_transform = '\n'.join(trunc)
    return source_transform


def decorate_transform(source_input, clique_by_type, max_toks=8192):
    source_transform = decorate_set_of_ents(source_input, rel_spans=rel_spans, add_space=True)

    num_problems = len(clique_by_type['problems'])
    num_treatments = len(clique_by_type['treatments'])
    num_tests = len(clique_by_type['tests'])

    # Use only 1 word-piece token
    source_transform = re.sub(r'\s?<e>', ' {{', source_transform)
    source_transform = re.sub(r'</e>\s?', '}} ', source_transform)
    guidance = f'\n\n### ENTITIES\nPROBLEMS: {num_problems}\nTREATMENTS: {num_treatments}\nTESTS: {num_tests}'

    source_transform = source_transform.replace('<doc-sep>', '')

    clique_source_toks = len(word_tokenize(source_transform))
    if args.filter_clique_notes or clique_source_toks > max_toks or not curr_clique['add_unused_to_queue']:
        print(f'Shrinking input from {clique_source_toks} for just in clique-entity lines')
        source_transform = remove_non_ents(source_transform).strip()  # , clique_cluster_span_set)

        clique_source_toks = len(word_tokenize(source_transform))
        if clique_source_toks > max_toks:
            print(f'{clique_source_toks} > {max_toks}. Going to take most recent lines')
            source_transform = take_latest(source_transform, max_toks=max_toks)
    source_transform = re.sub(r'\n{2,}', '\n\n', source_transform).strip()
    source_transform = re.sub(r'({{ ){2,}', '{{ ', source_transform)
    source_transform = re.sub(r'(}} ){2,}', '}} ', source_transform)
    return source_transform, guidance


def frost_transform(source_input, clique_by_type, max_toks=8192):
    source_transform = source_input

    problem_str = '; '.join([x[0] for x in clique_by_type['problems']])
    treatment_str = '; '.join([x[0] for x in clique_by_type['treatments']])
    test_str = '; '.join([x[0] for x in clique_by_type['tests']])

    guidance = f'\n\n### ENTITIES\nPROBLEMS: {problem_str}\nTREATMENTS: {treatment_str}\nTESTS: {test_str}'
    source_transform = source_transform.replace('<doc-sep>', '')
    clique_source_toks = len(word_tokenize(source_transform))

    if args.filter_clique_notes or clique_source_toks > max_toks or not curr_clique['add_unused_to_queue']:
        source_transform = remove_non_ents_from_list(source_transform, rel_spans).strip()
        clique_source_toks = len(word_tokenize(source_transform))
        if clique_source_toks > max_toks:
            print(f'{clique_source_toks} > {max_toks}. Going to take most recent lines')
            source_transform = take_latest(source_transform, max_toks=max_toks)
    return source_transform, guidance


def remove_duplicates_preserve_order(arr):
    """
    Removes duplicates from a list while preserving order.

    :param arr: A list with possible duplicates.
    :return: A new list with duplicates removed, in the same order as the original list.
    """
    seen = set()
    result = []
    for item in arr:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def get_attr(tag, attr):
    return re.search(r'\s' + attr + r'=([^ ]+)', tag).group(1).strip('<>: ')


def embed_concept_spans(model, tokenizer, syns, batch_size=4096, verbose=False):
    all_reps = []
    batch_starts = np.arange(0, len(syns), batch_size)
    num_batches = len(batch_starts)
    batch_ct = 0
    for i in batch_starts:
        batch_ct += 1
        toks = tokenizer.batch_encode_plus(
            syns[i:i + batch_size], padding='max_length', max_length=25, truncation=True, return_tensors='pt')
        toks_cuda = {k: v.to(model.device) for k, v in toks.items()}
        with torch.no_grad():
            output = model(**toks_cuda)
            cls_rep = output[0][:, 0, :]
            all_reps.append(cls_rep.cpu().detach().numpy())
        if verbose:
            print(f'{batch_ct}/{num_batches}')
    return np.concatenate(all_reps, axis=0)


def transform_text_for_llama(
        html_str, include_header=True, include_title=True, include_sent_markers=False, meta=None
):
    tps = html_str.split('<SEP>')
    curr_str = ''
    note_idx = -1
    for idx, tp in enumerate(tps):
        if tp.startswith('<d') and include_title:
            if idx > 0 and tps[idx - 1].startswith('<d'):
                continue
            try:
                curr_note_id = get_attr(tp, 'note_id')
                note_title = ' '.join(map(str.capitalize, curr_note_id.split('-')[9:]))
            except:
                title = get_attr(tp, 'title')
                note_title = ' '.join(map(str.capitalize, title.split('_')))
            note_idx += 1
            curr_str += '\n\n### Title: ' + note_title + '\n'
            if meta is not None:
                note_meta = meta[note_idx]
                curr_str += note_meta + '\n'
        elif tp.startswith('<h'):
            raw = get_attr(tp, 'raw')
            if raw.lower() == 'unknown':
                continue
            raw_section = re.sub(r'[_\s]+', ' ', raw).strip()
            if len(curr_str) > 0 and curr_str[-1] != '\n':
                curr_str += '\n'
            if include_header:
                curr_str += raw_section + ':\n'
        elif idx > 0 and tps[idx - 1].startswith('<s'):
            # sent_str = remove_tags_from_sent(tp)
            if len(curr_str) > 0 and curr_str[-1] not in {'\n', '\t', ' '}:
                curr_str += ' '
            if include_sent_markers:
                curr_str += '<s>' + tp + '</s>'
            else:
                curr_str += tp
    return curr_str.strip()


def get_entity_guidance(
        example_id, all_ent_probs, source_ent_clusters, source_ent_types, max_csize=15,
        pred_ent_threshold=_DEFAULT_PRED_ENT_THRESHOLD, min_ents=3, max_ents=100
):
    # priority = np.argsort(-np.array(ent_probs))
    ent_probs = [x for x in all_ent_probs if x['example_id'] == example_id][0]

    pred_idxs = get_pred_ent_cluster_idxs(ent_probs['cluster_pred_probs'], pred_ent_threshold, min_ents, max_ents)

    pred_source_clusters = [source_ent_clusters[i] for i in pred_idxs]
    pred_source_types = [source_ent_types[i] for i in pred_idxs]
    target_problems = [c for c, t in zip(pred_source_clusters, pred_source_types) if t == 'PROBLEM']
    target_tests = [c for c, t in zip(pred_source_clusters, pred_source_types) if t == 'TEST']
    target_treatments = [c for c, t in zip(pred_source_clusters, pred_source_types) if t == 'TREATMENT']

    ents = {
        'problems': target_problems,
        'treatments': target_treatments,
        'tests': target_tests,
    }

    problems = '\n'.join(['; '.join(z[:min(len(z), max_csize)]) for z in target_problems])
    treatments = '\n'.join(['; '.join(z[:min(len(z), max_csize)]) for z in target_treatments])
    tests = '\n'.join(['; '.join(z[:min(len(z), max_csize)]) for z in target_tests])
    guidance = f'# PROBLEMS:\n{problems}\n\n# TREATMENTS:\n{treatments}\n\n# TESTS:\n{tests}'
    return guidance, ents, pred_source_clusters


def load_ent_embeds():
    print('Loading embeddings...')
    with open('/nlp/projects/summarization/bhc_data_cleanup/entity_embeddings_test.pk', 'rb') as fd:
        span2embed = pickle.load(fd)
    return span2embed


def load_ent_info(args, example_id, span2embed):
    ent_suffix = '' if args.dataset == 'epic' else f'_{args.dataset}'
    entity_fn = os.path.join(IN_DIR, f'entity_stanza{ent_suffix}', f'{example_id}.json')
    entity_merge_fn = os.path.join(IN_DIR, f'entity_stanza{ent_suffix}_top_ents', f'{example_id}.json')

    with open(entity_merge_fn, 'r') as fd:
        ent_merges = ujson.load(fd)

    source_ent_clusters = ent_merges['source_cluster_spans']
    source_ent_types = ent_merges['source_cluster_types']

    with open(entity_fn, 'r') as fd:
        ents = ujson.load(fd)
        target_ents = ents['target']

        target_span_cts = Counter([x['text'] for x in target_ents])
        target_spans = list(sorted(list(target_span_cts.keys())))
        target_embeds = np.array([span2embed[span] for span in target_spans])

        source_ents = ents['source']
        source_span_cts = Counter([x['text'] for x in source_ents])
        source_spans = list(sorted(list(source_span_cts.keys())))
        source_embeds = np.array([span2embed[span] for span in source_spans])

    with open(entity_merge_fn, 'r') as fd:
        merges = ujson.load(fd)

    return {
        'ent_merges': merges,
        'source_ent_clusters': source_ent_clusters,
        'source_ents': source_ents,
        'source_ent_types': source_ent_types,
        'source_embeds': source_embeds,
        'target_embeds': target_embeds,
    }


def generate_note_meta(note_tag, note_idx, num_notes, admit_date, discharge_date, now=None):
    if now is None:
        now = datetime.strptime(get_attr(note_tag, 'time').split('_')[0], "%m-%d-%y").date()
    days_into_stay = (now - admit_date).days
    los = (discharge_date - admit_date).days
    is_day_of_admission = now == admit_date
    is_day_of_discharge = now == discharge_date
    now_str = now.strftime("%m/%d/%Y")
    meta = f"DATE: {now_str}\n\nNOTE ORDER: {note_idx + 1} of {num_notes}\n\nDAY: {days_into_stay} of {los}"

    if is_day_of_admission:
        meta += "\n\nON DAY OF ADMISSION"
    if is_day_of_discharge:
        meta += "\n\nON DAY OF DISCHARGE"
    return meta


def remove_duplicates(arr):
    seen = set()
    covered_toks = set()
    new_arr = []
    for sent in arr:
        sent_toks = list(set([x.strip().lower() for x in re.split('\W+', sent) if x.strip().lower() not in BHC_STOPWORDS and len(x.strip()) > 0]))
        num_new = len([
            tok for tok in sent_toks if tok not in covered_toks
        ])
        if num_new < 2 or sent.lower() in seen:
            continue
        for tok in sent_toks:
            covered_toks.add(tok)
        new_arr.append(sent)
        seen.add(sent.lower())
    return arr


def extract_target_ents(target_sents, nlp):
    concepts = []
    for target_idx, sent in enumerate(target_sents):
        ents = nlp(sent).entities
        ents = [{'text': ent.text, 'type': ent.type} for ent in ents]
        for ent in ents:
            ent.update({'sent_idx': target_idx})
            concepts.append(ent)
    return concepts


def get_pred_ent_cluster_idxs(cluster_pred_probs, pred_ent_threshold, min_ents, max_ents):
    priority = np.argsort(-np.array(cluster_pred_probs)).tolist()
    probs_sorted = [cluster_pred_probs[j] for j in priority]

    pred_idxs = [idx for idx, score in zip(priority, probs_sorted) if score >= pred_ent_threshold]

    if len(pred_idxs) < min_ents:
        pred_idxs = priority[:min_ents]
    elif len(pred_idxs) > max_ents:
        pred_idxs = priority[:max_ents]
    return pred_idxs


def extract_pred_ent_span_set(
        ent_probs, ent_info, pred_ent_threshold=_DEFAULT_PRED_ENT_THRESHOLD, min_ents=3, max_ents=100
):
    pred_idxs = get_pred_ent_cluster_idxs(ent_probs['cluster_pred_probs'], pred_ent_threshold, min_ents, max_ents)

    pred_source_ent_set = set()
    for x in [ent_info['source_ent_clusters'][i] for i in pred_idxs]:
        for y in x:
            pred_source_ent_set.add(y)
    return pred_source_ent_set


def get_inference_cliques(pred_cluster_spans, source_ents):
    source_ents_in_clusters = []
    full_sent_idx_to_cluster_idx = defaultdict(set)
    for cidx, cluster in enumerate(pred_cluster_spans):
        matching_ents = [
            ent for ent in source_ents if ent['text'] in cluster
        ]
        for ent in matching_ents:
            full_sent_idx_to_cluster_idx[ent['full_sent_idx']].add(cidx)
        source_ents_in_clusters.append(matching_ents)

    num_nodes = len(pred_cluster_spans)
    source_graph = Graph(num_nodes)

    if num_nodes == 1:
        component_idxs = [[0]]
    else:
        for k, edge_set in full_sent_idx_to_cluster_idx.items():
            edge_arr = list(sorted(list(edge_set)))
            for i, j in itertools.combinations(edge_arr, 2):
                source_graph.addEdge(i, j)

        component_idxs = source_graph.connectedComponents()

    clique_ents = [
        [source_ents_in_clusters[idx] for idx in idxs] for idxs in component_idxs
    ]

    clique_clusters = [
        [pred_cluster_spans[idx] for idx in idxs] for idxs in component_idxs
    ]

    clique_notes = []
    for clique_ent in clique_ents:
        notes = set()
        for ce in clique_ent:
            for c in ce:
                notes.add(c['note_idx'])

        notes = list(sorted(list(notes)))
        clique_notes.append(notes)
    return {
        'clique_notes': clique_notes,
        'clique_cluster_spans': clique_clusters,
        'clique_ents': clique_ents,
    }


def precise_decorate(span, text, add_space=False):
    space = ' ' if add_space else ''
    escaped = re.escape(span)
    tps = re.split(rf'(\W{escaped}\W)', text, flags=re.IGNORECASE)
    updated = ''
    for tp in tps:
        match = re.search(rf'\W({escaped})\W', tp, flags=re.IGNORECASE)
        if match is None:
            updated += tp
        else:
            start, end = match.start(1), match.end(1)
            updated += tp[:start] + f'<e>{space}' + tp[start:end] + f'{space}</e>' + tp[end:]

    if updated.startswith(span):
        return (f'<e>{space}' + span + f'{space}</e> ' + updated.lstrip(span)).strip()
    if updated.endswith(span):
        return updated.rstrip(span) + f'{space}<e>{space}' + span + f'{space}</e>'

    return updated


def decorate_set_of_ents(source, rel_spans, add_space=False):
    decorated_source = source

    rel_spans_lower = list(set([x.strip(string.punctuation).lower() for x in list(rel_spans)]))
    rel_spans_lower = list(sorted(rel_spans_lower, key=lambda x: -len(x)))

    for span in list(rel_spans_lower):
        decorated_source = precise_decorate(span, decorated_source, add_space=add_space)
    return decorated_source


def decorate_clique_src(source, target=None, merges=None, rel_spans=None):
    if rel_spans is None:
        rel_spans = set()
        for tgt_ent, src_ents in merges['t2s_alignments'].items():
            if tgt_ent in target:
                for src_ent in src_ents:
                    rel_spans.add(src_ent)

    return decorate_set_of_ents(source, rel_spans=rel_spans)


def generate_clique_input(notes, note_idxs, admit_date=None, discharge_date=None):
    num_notes = len(notes)
    outputs = []
    for note_idx in note_idxs:
        note = notes[note_idx]
        tags = note.split('<SEP>')
        if tags[0] == tags[1]:
            note = '<SEP>'.join(tags[1:])

        if admit_date is None:
            note_str = transform_text_for_llama(
                note, include_header=True, include_title=True, include_sent_markers=False
            )
        else:
            meta = generate_note_meta(
                tags[0], note_idx, num_notes, admit_date=admit_date, discharge_date=discharge_date
            )
            note_str = transform_text_for_llama(
                note, include_header=True, include_title=True, include_sent_markers=False, meta=[meta]
            )
        note_str = note_str.replace('?', ' ').replace(u'\xa0', ' ')
        outputs.append(note_str)
    return '\n\n'.join(outputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('BHC Partial Summarization.')
    parser.add_argument('--data_dir', default='/nlp/projects/summarization/bhc_data_cleanup')

    parser.add_argument('--dataset', default='epic')
    parser.add_argument('--experiment', default='clique_unlike')

    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('-debug', default=False, action='store_true')

    parser.add_argument('--max_examples', default=1000, type=int)

    # Clique parameter
    parser.add_argument('-compress_cliques', default=False, action='store_true')
    parser.add_argument('-filter_clique_notes', default=False, action='store_true')

    parser.add_argument('--split', default='test')
    parser.add_argument('-human', default=False, action='store_true')
    parser.add_argument('--pred_ent_threshold', default=0.75, type=float)
    parser.add_argument('-overwrite', default=False, action='store_true')

    # Llama Arguments
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--max-prompt-tokens', type=int, default=8192)
    parser.add_argument('--repetition-penalty', type=float, default=1.0)
    parser.add_argument('--ckpt', default='latest')

    args = add_args(parser).parse_args()

    # Model loading part
    is_frost = 'frost' in args.experiment
    print('\n\nTHIS IS A FROSTY MODEL.\n\n')
    args.model = f'/nlp/projects/summarization/bhc_data_cleanup/bhc_weights/yarn-7b-8k-{args.experiment}'
    print(f'Loading model from {args.model}')
    args.yarn = 16.0
    args.finetuned = True
    args.original_max_position_embeddings = 4096
    args.flash_attention = True
    args.custom_model_together = True

    tokenizer_model = 'NousResearch/Llama-2-7b-hf'  # Switched from args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, model_max_length=sys.maxsize, trust_remote_code=True)

    if args.ckpt == 'final':
        model_path = args.model
    else:
        # Use latest checkpoint
        ckpts = [x for x in os.listdir(args.model) if 'step' in x]
        print(f'Checkpoints:\n', '\n'.join(ckpts))
        if args.ckpt == 'latest':
            steps = [int(x.split('_')[1]) for x in ckpts]
            ckpt_idx = int(np.argmax(steps))
        else:
            ckpt_idx = int([x.split('_')[1] for x in ckpts].index(args.ckpt))

        model_path = os.path.join(args.model, ckpts[ckpt_idx])
    print(f'Loading from {model_path}...')
    model = load_model_and_apply_patches(model_path, args)

    pipe = pipeline(
        'text-generation', model=model, tokenizer=tokenizer,  # pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
        # repetition_penalty=args.repetition_penalty,
    )

    print('Loading SAPBERT')
    sapbert_tokenizer = AutoTokenizer.from_pretrained(SAP_BERT)
    sapbert_model = AutoModel.from_pretrained(SAP_BERT).eval().to(args.device)
    nlp = stanza.Pipeline('en', package='mimic', processors={'ner': 'i2b2'}, use_gpu=True)
    rouge = load('rouge', keep_in_memory=True)

    tools = {
        'rouge': rouge,
        'nlp': nlp,
        'sapbert_model': sapbert_model,
        'sapbert_tokenizer': sapbert_tokenizer,
    }
    # End of loading tools
    if args.filter_clique_notes:
        out_dir = os.path.join(model_path, 'clique_filt')
    else:
        out_dir = os.path.join(model_path, 'clique')
    out_fn = f'{out_dir}.csv'
    os.makedirs(out_dir, exist_ok=True)

    if args.dataset == 'epic':
        if args.human:
            ent_fn = os.path.join(IN_DIR, 'bhc_weights', 'fixed', 'test_human.json')
        else:
            ent_fn = os.path.join(IN_DIR, 'bhc_weights', 'fixed', 'test_test.json')
    elif args.dataset == 'cumc':
        ent_fn = os.path.join(IN_DIR, 'bhc_weights', 'fixed', 'test_cumc_test.json')
    else:
        assert args.dataset == 'mimic'
        ent_fn = os.path.join(IN_DIR, 'bhc_weights', 'fixed', 'test_mimic_test.json')

    with open(ent_fn, 'r') as fd:
        all_ent_probs = ujson.load(fd)

    print('Reading in dataset...')
    visit_meta = {}
    if args.dataset == 'cumc':
        data_dir = f'/nlp/projects/summarization/bhc_data_cleanup/cumc_test'
        print(f'Reading in data from {data_dir}')
        data = load_from_disk(data_dir)
    elif args.dataset == 'epic':
        data_dir = '/nlp/projects/summarization/bhc_data_cleanup/summarization_deduped_dataset'
        visit_meta = pd.read_csv('/nlp/projects/summarization/bhc_data_cleanup/bhc_test_meta.csv')
        visit_meta = {
            row['visit_id']: row for row in visit_meta.to_dict('records')
        }
        print(f'Reading in data from {data_dir}')
        data = load_from_disk(data_dir)[args.split]
    else:
        data_dir = '/nlp/projects/summarization/bhc_data_cleanup/mimic_test_filt'
        print(f'Reading in data from {data_dir}')
        data = load_from_disk(data_dir)

    if args.dataset == 'epic' and args.split == 'test':
        if args.human:
            valid_visit_ids = set(map(str, pd.read_csv(
                '/nlp/projects/summarization/bhc_data_cleanup/bhc_human_meta.csv'
            )['visit_id']))
        else:
            valid_visit_ids = set(map(str, pd.read_csv(
                '/nlp/projects/summarization/bhc_data_cleanup/bhc_test_meta.csv'
            )['visit_id']))
        data = data.filter(
            lambda row: row['visit_id'] in valid_visit_ids
        )
        print(f'{len(data)} examples after removing contaminated samples.')

    n = len(data)
    if args.max_examples is not None and args.max_examples < n:
        idxs = list(sorted(np.random.choice(np.arange(n), size=(args.max_examples), replace=False)))
        data = data.select(idxs)

    example_ids = set([x['example_id'] for x in all_ent_probs])
    prev = len(data)
    data = data.filter(lambda row: row['example_id'] in example_ids)
    new = len(data)
    print(f'Entity Probabilities for {new} / {prev} examples. Filtering...')

    span2embed = load_ent_embeds()

    scores = []
    outputs = []

    for example in tqdm(data):
        example_id = example['example_id']
        save_fn = os.path.join(out_dir, f'{example_id}.json')

        if os.path.exists(save_fn) and not args.overwrite:
            print(f'Already exists --> {save_fn}. Skipping...')
            with open(save_fn, 'r') as fd:
                out_row = ujson.load(fd)
                scores.append(out_row.copy())
                outputs.append(out_row.copy())
            continue

        # Entity Stuff
        ent_probs = [x for x in all_ent_probs if x['example_id'] == example_id][0]
        ent_info = load_ent_info(args, example_id, span2embed)
        guidance, ents_in_guidance, pred_source_clusters = get_entity_guidance(
            example_id, all_ent_probs, ent_info['source_ent_clusters'], ent_info['source_ent_types'],
            pred_ent_threshold=args.pred_ent_threshold
        )

        target_no_dup = '\n'.join(remove_duplicates_preserve_order(example['target_sents']))

        pred_source_ent_set = extract_pred_ent_span_set(
            ent_probs, ent_info, pred_ent_threshold=args.pred_ent_threshold
        )

        notes = split_into_notes(example['source'])

        # Create cliques from pred_source_clusters
        cliques = get_inference_cliques(pred_source_clusters, ent_info['source_ents'])
        queue = []
        for clique_idx in range(len(cliques['clique_notes'])):
            queue.append({
                'id': clique_idx,
                'clique_note_idxs': cliques['clique_notes'][clique_idx],
                'clique_cluster_spans': cliques['clique_cluster_spans'][clique_idx],
                'add_unused_to_queue': True,
            })

        ignored_sents = []
        pred_sents = []
        clique_order = []
        all_clique_embeds = []
        all_uncovered_clique_clusters = []
        all_covered_clique_clusters = []
        prev_pred_ent_embeds = []

        generated_by_type = {
            'problems': set(),
            'treatments': set(),
            'tests': set(),
        }

        while len(queue) > 0:
            curr_clique = queue.pop(0)

            clique_order.append(curr_clique)
            clique_note_idxs = curr_clique['clique_note_idxs']
            clique_cluster_spans = curr_clique['clique_cluster_spans']
            clique_embeds = [
                embed_concept_spans(
                    tools['sapbert_model'], tools['sapbert_tokenizer'], span
                ) for span in clique_cluster_spans
            ]

            clique_by_type = {'problems': [], 'treatments': [], 'tests': []}
            for cluster in curr_clique['clique_cluster_spans']:
                for cluster_idx in range(len(ent_info['ent_merges']['source_cluster_spans'])):
                    if cluster == ent_info['ent_merges']['source_cluster_spans'][cluster_idx]:
                        ctype = ent_info['ent_merges']['source_cluster_types'][cluster_idx].lower() + 's'
                        clique_by_type[ctype].append(cluster)

            # Remove any guidance cliques which are already included in prev_ent_embeds
            if len(prev_pred_ent_embeds) > 0:
                clique_redundancies = [
                    cosine_similarity(np.concatenate(prev_pred_ent_embeds), e) for e in clique_embeds
                ]

                clique_is_redundant = [np.any(x >= ENT_MERGE_THRESHOLD) for x in clique_redundancies]

                clique_cluster_spans = [
                    s for s, red in zip(clique_cluster_spans, clique_is_redundant) if not red
                ]
                clique_embeds = [
                    s for s, red in zip(clique_embeds, clique_is_redundant) if not red
                ]

            if len(clique_cluster_spans) == 0:
                print('All clique spans were previously covered. Breaking out.')
                continue

            for e in clique_embeds:
                all_clique_embeds.append(e)

            clique_cluster_is_covered = [False for _ in range(len(clique_cluster_spans))]

            c_pred = []
            admit_date = discharge_date = None
            if 'first_date' in example:
                admit_date = datetime.strptime(example['first_date'].split('_')[0], "%m-%d-%y").date()
                discharge_date = datetime.strptime(example['last_date'].split('_')[0], "%m-%d-%y").date()

            source_input = generate_clique_input(notes, clique_note_idxs, admit_date, discharge_date)
            clique_cluster_span_set = set(list(itertools.chain(*clique_cluster_spans)))

            rel_spans = set(
                list(itertools.chain(*curr_clique['clique_cluster_spans']))
            )

            if is_frost:
                source_transform, guidance = frost_transform(
                    source_input, clique_by_type, max_toks=args.max_prompt_tokens
                )
            else:
                source_transform, guidance = decorate_transform(
                    source_input, clique_by_type, max_toks=args.max_prompt_tokens
                )

            clique_guidance = ents_in_guidance.copy()
            for k, arr in clique_guidance.items():
                clique_guidance[k] = [x for x in arr if x in clique_cluster_spans]

            prompt = source_transform + guidance + BHC_PREFIX

            cp = gen_from_clique(pipe=pipe, prompt=prompt)
            cp = cp.split('\n')

            num_pred_ents_for_clique = 0
            for c in cp:
                cpse = extract_target_ents([c], tools['nlp'])
                cpse_spans = [x['text'] for x in cpse]
                num_pred_ents_for_clique += len(cpse)
                if len(cpse_spans) == 0:
                    cpse_embeds = []
                else:
                    cpse_embeds = embed_concept_spans(
                        tools['sapbert_model'], tools['sapbert_tokenizer'], cpse_spans
                    )

                def unigram_repetitive(text, max_unigram_rep=10):
                    tok_cts = Counter(
                        [x for x in text.lower().split(' ') if len(x) > 0 and x not in BHC_STOPWORDS]
                    )
                    return tok_cts.most_common()[0][1] >= max_unigram_rep

                hits = 0
                if len(cpse_embeds) == 0:
                    print(f'Sentence below has NO entities. Not adding to summary.')
                    print(c)
                elif unigram_repetitive(c):
                    print(f'Repeated sentence -> {c}. Will not include.')
                else:
                    for tmp_clique_idx in range(len(clique_embeds)):
                        sim_mat = cosine_similarity(cpse_embeds, clique_embeds[tmp_clique_idx])
                        if np.any(sim_mat >= ENT_MERGE_THRESHOLD):
                            hits += 1
                            clique_cluster_is_covered[tmp_clique_idx] = True

                if hits == 0:
                    print(f'Sentence below covers NO entities from guidance. Not adding to summary.')
                    print(c)
                    ignored_sents.append(c)

                if c not in pred_sents and hits > 0:
                    pred_sents.append(c)
                    for ent in cpse:
                        generated_by_type[ent['type'].lower() + 's'].add(ent['text'])
                    c_pred.append(c)
                    prev_pred_ent_embeds.append(cpse_embeds)

            # Extract entities predicted and compare to the ones in the guidance
            # If > 0, create a clique out of the unused
            assert len(clique_cluster_spans) == len(clique_cluster_is_covered)
            uncovered_clique_clusters = [
                cluster for cluster, is_covered in
                zip(clique_cluster_spans, clique_cluster_is_covered) if not is_covered
            ]

            covered_clique_clusters = [
                cluster for cluster, is_covered in
                zip(clique_cluster_spans, clique_cluster_is_covered) if is_covered
            ]

            all_covered_clique_clusters.append(covered_clique_clusters)

            clique_cov_frac = sum(clique_cluster_is_covered) / len(clique_cluster_is_covered)
            clique_precision = 0 if num_pred_ents_for_clique == 0 else (
                    sum(clique_cluster_is_covered) / num_pred_ents_for_clique
            )

            print(f'Clique Coverage -> {clique_cov_frac}')
            print(f'Clique Precision -> {clique_precision}')

            if len(uncovered_clique_clusters) > 0 and curr_clique['add_unused_to_queue']:
                cluster_to_add = {
                    'add_unused_to_queue': False,
                    'id': str(curr_clique['id']) + '_leftover',
                    'clique_cluster_spans': uncovered_clique_clusters,
                    'clique_note_idxs': curr_clique['clique_note_idxs'],
                }
                print(f'Missed {len(uncovered_clique_clusters)} clusters. Adding to new cluster to try again.')
                queue.insert(0, cluster_to_add)
            elif len(uncovered_clique_clusters) > 0 and not curr_clique['add_unused_to_queue']:
                print('Failed twice to cover a cluster. Adding them to unused queue.')
                all_uncovered_clique_clusters.append(uncovered_clique_clusters)

        pred_sents = remove_duplicates(pred_sents)
        if len(pred_sents) == 0:
            print('0 predicted sentences. Using ignored one(s).')
            pred_sents = ignored_sents

        prediction = '\n'.join(pred_sents)

        if args.compress_cliques:
            generated_by_type = {
                k: [[x] for x in list(v)] for k, v in generated_by_type.items()
            }
            _, final_guidance = frost_transform(
                prediction, generated_by_type, max_toks=args.max_prompt_tokens
            )
            final_prompt = prediction + final_guidance + BHC_FULL
            compressed = gen_from_clique(pipe=pipe, prompt=final_prompt)
            prediction = compressed

        print('\n\n')
        print(prediction)
        print('\n\n')

        out_row = {
            'example_id': example['example_id'],
            'reference': target_no_dup,
            'prediction': prediction
        }

        v_meta = visit_meta.get(example.get('visit_id', ''), {})
        out_row.update(v_meta)

        print(f'Saving to {save_fn}')
        with open(save_fn, 'w') as fd:
            json.dump(out_row, fd)

        outputs.append(out_row)

    outputs = pd.DataFrame(outputs)

    print(f'Saving predictions to {out_fn}...')
    outputs.to_csv(out_fn, index=False)

    print(outputs.select_dtypes(include='number').mean())
