"""
Pre-process the Yelp point of interest recommendations dataset.
"""
import argparse
import json, codecs
import collections
import os
import pickle
import pprint
import re
import statistics
import time
import sklearn.feature_extraction.text as skfeat
import numpy as np
from sklearn import neighbors
from sentence_transformers import SentenceTransformer, models

import langid
import openai

from . import yelppoi_prompts
from . import data_utils


def print_dataset(in_path):
    """
    - Read the infoneeds.json
    - Read the relevance ratings.
    - Print the rated points of interest.
    """
    # Read the need.
    with codecs.open(os.path.join(in_path, 'infoneeds.json'), 'r', 'utf-8') as fp:
        qid2need = json.load(fp)
    print(f'Queries: {len(qid2need)}')
    
    # Read the relevance file.
    with codecs.open(os.path.join(in_path, 'relevance/qrels.trec'), 'r', 'utf-8') as fp:
        qid2rated_pool = collections.defaultdict(list)
        for line in fp:
            qid, _, candid, rel = line.strip().split()
            qid2rated_pool[qid].append((candid, int(rel)))
    print(f'Rated pools: {len(qid2rated_pool)}')
    
    # Read the candidates.
    cands = dict()
    for root_dir, sub_dirs, filenames in os.walk(os.path.join(in_path, 'poi_dataset')):
        for fname in filenames:
            with codecs.open(os.path.join(root_dir, fname), 'r', 'utf-8') as fp:
                city_pois = json.load(fp)
                cands.update(city_pois)
    print(f'Cands read: {len(cands)}')
                
    # Print the query and candidates.
    out_dir = os.path.join(in_path, 'readable_judgements')
    data_utils.create_dir(out_dir)
    qid2missingcand = collections.defaultdict(int)
    qmetamissing = collections.defaultdict(int)
    total_chars = 0
    for qid, needd in qid2need.items():
        outf = codecs.open(os.path.join(out_dir, f'{qid}.txt'), 'w', 'utf-8')
        try: outf.write('City: {:s}\n'.format(needd['City']))
        except TypeError: qmetamissing['City'] += 1
        try: outf.write('Country: {:s}\n'.format(needd['Country']))
        except TypeError:
            qmetamissing['Country'] += 1
        try: outf.write('Main category: {:s}\n'.format(needd['Main Category']))
        except TypeError:
            qmetamissing['Main Category'] += 1
        try: outf.write('Sub category: {:s}\n'.format(' - '.join(needd['Sub Categories'])))
        except TypeError:
            qmetamissing['Sub Categories'] += 1
        try: outf.write('Description: {:s}\n'.format(needd['Description']))
        except TypeError:
            qmetamissing['Description'] += 1
        outf.write('Rated POIs: {:}\n'.format(len(qid2rated_pool[qid])))
        rel_counts = collections.Counter([t[1] for t in qid2rated_pool[qid]])
        outf.write('Rel distr: {:}\n'.format(sorted(rel_counts.items())))
        outf.write('=================================\n')
        cand_cities = set()
        for candid, cand_rel in sorted(qid2rated_pool[qid], key=lambda t:t[1], reverse=True):
            try:
                candd = cands[candid]
                outf.write('REL: {:}\n'.format(cand_rel))
                outf.write('POI Name: {:}\n'.format(candd['name']))
                outf.write('POI City: {:}\n'.format(candd['city']))
                cand_cities.add(candd['city'])
                outf.write('POI Country: {:}\n'.format(candd['country_code']))
                outf.write('POI Main category: {:}\n'.format(candd['main_category']))
                outf.write('POI Sub category: {:}\n'.format(candd['sub_categories']))
                total_chars += sum([len(s['title'])+len(s['snippet']) for s in candd['snippets']])
                snippets = '\n'.join(['title: {:}\nsnippet: {:}'.
                                     format(s['title'], s['snippet'].strip()) for s in candd['snippets']])
                outf.write('Snippets:\n')
                outf.write('{:}\n\n'.format(snippets))
            except KeyError:
                qid2missingcand[qid] += 1
        outf.write('Cand cities: {:}\n'.format(cand_cities))
        outf.close()
    print(f'Snippet characters: {total_chars}')
    print('Missing cands:')
    print(dict(qid2missingcand))
    print('Queries missing metadata:')
    print(dict(qmetamissing))
    

def pointrec_to_json(in_path, json_path, en_snippets=False):
    """
    - Read the raw data and write out json files needed to produce predictions:
        'abstracts-csfcube-preds.jsonl' and 'test-pid2anns-{dataset}.json'
    - This function creates candidates based on only the rated test collection
        not the whole poi_dataset.
    """
    # Read the need.
    with codecs.open(os.path.join(in_path, 'infoneeds.json'), 'r', 'utf-8') as fp:
        qid2need = json.load(fp)
    print(f'Queries: {len(qid2need)}')
    qid2queries = dict()
    for qid, need in qid2need.items():
        try:
            subcats = ', '.join(need['Sub Categories']).lower()
            meta_description = 'Looking for {:s}, specifically {:s} in {:s}, {:s}'. \
                format(need['Main Category'].lower(), subcats, need['City'], need['Country'])
            subcat = need['Sub Categories']
        except TypeError:
            meta_description = 'Looking for {:s} in {:s}, {:s}'. \
                format(need['Main Category'].lower(), need['City'], need['Country'])
            subcat = None
        # Skipping the title.
        query = '{:s}. {:s}.'.format(meta_description, need['Description'])
        qid2queries[qid] = {
            'query': query,
            'city': need['City'],
            'state': need['State'],
            'country': need['Country'],
            'category': need['Main Category'].lower(),
            'subcategory': subcat
        }
    with codecs.open(os.path.join(json_path, 'pointrec_queries.json'), 'w', 'utf-8') as fp:
        json.dump(qid2queries, fp, indent=2)
        print(f'Write: {fp.name}')

    # Read the relevance file.
    with codecs.open(os.path.join(in_path, 'relevance/qrels.trec'), 'r', 'utf-8') as fp:
        qid2rated_pool = collections.defaultdict(list)
        for line in fp:
            qid, _, candid, rel = line.strip().split()
            qid2rated_pool[qid].append((candid, int(rel)))
    qid2test_pool = dict()
    uniq_cands = set()
    num_cands = []
    for qid, rated_pool in qid2rated_pool.items():
        cands = [t[0] for t in rated_pool]
        uniq_cands.update(cands)
        relevances = [t[1] for t in rated_pool]
        qid2test_pool[qid] = {'cands': cands, 'relevance_adju': relevances}
        num_cands.append(len(cands))
    print(f'Rated pools: {len(qid2rated_pool)}')
    assert(len(qid2queries) == len(qid2test_pool))
    print('Pool depths; mean: {:}, max: {:}, min: {:}'.
          format(statistics.mean(num_cands), max(num_cands), min(num_cands)))
    with codecs.open(os.path.join(json_path, 'test-pid2anns-pointrec.json'), 'w', 'utf-8') as fp:
        json.dump(qid2test_pool, fp)
        print(f'Write: {fp.name}')

    # Read the candidates.
    cid2cand_meta = dict()
    for root_dir, sub_dirs, filenames in os.walk(os.path.join(in_path, 'poi_dataset')):
        for fname in filenames:
            with codecs.open(os.path.join(root_dir, fname), 'r', 'utf-8') as fp:
                city_pois = json.load(fp)
                cid2cand_meta.update(city_pois)
    print(f'Cands read: {len(cid2cand_meta)}')
    candid2cands = dict()
    missing_cands = 0
    missing_snippets = 0
    cands_with_en_snippets = 0
    for candid in uniq_cands:
        try:
            candd = cid2cand_meta[candid]
        except KeyError:
            candd = None
            missing_cands += 1
        if candd:
            poi_name = candd['name']
            poi_city = candd['city']
            poi_country = candd['country_code']
            poi_category = candd['main_category'].lower()
            try:
                poi_subcategory = ', '.join([s.strip() for s in candd['sub_categories'].split(',') if s.strip()]).lower()
            except KeyError:
                poi_subcategory = None
            if en_snippets:
                all_snippets = [snippet for snippet in candd['snippets'] if
                                langid.classify(snippet['snippet'])[0] == 'en']
                if len(all_snippets) > 0:
                    cands_with_en_snippets += 1
            else:
                all_snippets = candd['snippets']
            if all_snippets:
                snippets = '. '.join(['title: {:}, snippet: {:}'.
                                     format(s['title'], s['snippet'].strip()) for s in all_snippets[:5]])
            else:
                snippets = None
                missing_snippets += 1
            cand_descr = '{:s} in {:s}, {:s} is known for {:s}.'.format(poi_name, poi_city, poi_country, poi_category)
            if poi_subcategory:
                cand_descr += ' Specifically: {:s}.'.format(poi_subcategory)
            if snippets:
                cand_descr += ' {:s}.'.format(snippets)
        else:
            poi_name = None
            poi_city = None
            poi_country = None
            poi_category = None
            poi_subcategory = None
            cand_descr = 'Nothing is known about this place.'
            
        candid2cands[candid] = {
            'description': cand_descr,
            'city': poi_city,
            'country': poi_country,
            'category': poi_category,
            'subcategory': poi_subcategory
        }
    print(f'Candidates: {len(candid2cands)}, missing_cands: {missing_cands}, cands missing snippets: {missing_snippets},'
          f' cands with EN snippets: {cands_with_en_snippets}')

    if en_snippets:
        out_cand_fname = os.path.join(json_path, 'candid2cands-pointrec-ensnippets.json')
    else:
        out_cand_fname = os.path.join(json_path, 'candid2cands-pointrec.json')
    with codecs.open(out_cand_fname, 'w', 'utf-8') as fp:
        json.dump(candid2cands, fp, indent=2)
        print(f'Write: {fp.name}')


def get_pointrec_initial_cands_gold(in_path, json_path):
    """
    - Read the raw data and write out json files 'test-pid2anns-{dataset}.json'
        of pre-feteched candidates. The prefeteching is based on oracle information
        of which the category, subcategory, and the city, country of the request are
        marked as relevant.
    - This function creates candidates based on the whole poi_dataset.
    """
    # Read the need.
    with codecs.open(os.path.join(in_path, 'infoneeds.json'), 'r', 'utf-8') as fp:
        qid2need = json.load(fp)
    qid2queries = collections.OrderedDict()
    for qid, need in qid2need.items():
        qid2queries[qid] = {
            'city': need['City'],
            'state': need['State'],
            'country': need['Country'],
            'category': need['Main Category'].lower()
        }
    print(f'Queries: {len(qid2need)}')
    
    # Read the relevance file.
    with codecs.open(os.path.join(in_path, 'relevance/qrels.trec'), 'r', 'utf-8') as fp:
        qid2rated_pool = collections.defaultdict(dict)
        for line in fp:
            qid, _, candid, rel = line.strip().split()
            rel = int(rel)
            qid2rated_pool[qid][candid] = rel
    print(f'Rated pools: {len(qid2rated_pool)}')
    assert (len(qid2rated_pool) == len(qid2queries))
    
    # Read the candidates.
    city2cid = collections.defaultdict(set)
    cid2cand_meta = dict()
    read_cands = 0
    for root_dir, sub_dirs, filenames in os.walk(os.path.join(in_path, 'poi_dataset')):
        for fname in filenames:
            with codecs.open(os.path.join(root_dir, fname), 'r', 'utf-8') as fp:
                city_pois = json.load(fp)
                for cid, poi_meta in city_pois.items():
                    city2cid[poi_meta['city']].add(cid)
                    read_cands += 1
                cid2cand_meta.update(city_pois)
    print(f'Cands read: {read_cands}; Cities: {len(city2cid)}')
    candid2cands = collections.OrderedDict()
    for candid, candd in cid2cand_meta.items():
        poi_name = candd['name']
        poi_city = candd['city']
        poi_country = candd['country_code']
        poi_category = candd['main_category'].lower()
        try:
            poi_subcategory = ', '.join(
                [s.strip() for s in candd['sub_categories'].split(',') if s.strip()]).lower()
        except KeyError:
            poi_subcategory = None
        # Get a full description for use by ranking models.
        all_snippets = candd['snippets']
        if all_snippets:
            snippets = '. '.join(['title: {:}, snippet: {:}'.
                                 format(s['title'], s['snippet'].strip()) for s in all_snippets[:5]])
        else:
            snippets = None
        cand_descr = '{:s} in {:s}, {:s} is known for {:s}.'.format(poi_name, poi_city, poi_country, poi_category)
        if poi_subcategory:
            cand_descr += ' Specifically: {:s}.'.format(poi_subcategory)
        if snippets:
            cand_descr += ' {:s}.'.format(snippets)
        # Get a meta description for shortlisting candidates.
        cand_meta_descr = '{:s}, {:s} is known for {:s}.'.format(poi_city, poi_country, poi_category)
        if poi_subcategory:
            cand_meta_descr += ' Specifically: {:s}.'.format(poi_subcategory)
        candid2cands[candid] = {
            'description': cand_descr,
            'meta_description': cand_meta_descr,
            'city': poi_city,
            'country': poi_country,
            'category': poi_category,
            'subcategory': poi_subcategory
        }
    print(f'Candidates: {len(candid2cands)}')
    
    # Get the top candidates for each query.
    city_cands, citycat_cands = [], []
    city_recall, citycat_recall = [], []
    uniqcandids2cands_city = dict()
    uniqcandids2cands_citycat = dict()
    qid2test_pool_city = dict()
    qid2test_pool_citycat = dict()
    outf = codecs.open(os.path.join(json_path, f'candid2all_cands-recalls-citycat.txt'), 'w', 'utf-8')
    for qid, query_meta in qid2queries.items():
        # Handle for cities which dont map cleanly to one place or are noisy
        if query_meta['city'] == 'New York City':
            qcity = ['New York']
        elif query_meta['city'] == 'Walla walla':
            qcity = ['Walla Walla']
        elif query_meta['city'] == 'Zurich':
            qcity = ['Zürich']
        elif query_meta['city'] == 'Via San Vitale':
            qcity = ['Bologna']
        elif query_meta['city'] == 'Lucerne':
            qcity = ['Lucerne', 'Luzern', 'Zürich']
        elif query_meta['city'] == 'Stavanger':
            qcity = ['Stavanger', 'Oslo']
        elif query_meta['city'] == 'Coimbra':
            qcity = ['Coimbra', 'Lisbon']
        elif query_meta['city'] == 'Sintra':
            qcity = ['Sintra', 'Lisbon']
        elif query_meta['city'] == 'Vancouver':
            qcity = ['Vancouver', 'North Vancouver', 'West Vancouver']
        elif query_meta['city'] == 'Tokyo':
            qcity = ['Tokyo', 'Yokohama', 'Minato', 'Taitō', 'Shibuya', ]
        else:
            qcity = [query_meta['city']]
        qcategory = query_meta['category']
        # Gather the city and categories for relevants.
        rel_cids = [cid for cid, rel in qid2rated_pool[qid].items() if rel > 0]
        rel_cats = collections.defaultdict(int)
        rel_cities = collections.defaultdict(int)
        rel_countries = collections.defaultdict(int)
        for cid in rel_cids:
            try:
                rel_cats[candid2cands[cid]['category']] += 1
                rel_cities[candid2cands[cid]['city']] += 1
                rel_countries[candid2cands[cid]['country']] += 1
            except KeyError:
                rel_cats['Unknown'] += 1
                rel_cities['Unknown'] += 1
                rel_countries['Unknown'] += 1
        # Gather the pois for city or city+category match.
        all_city_pois, all_city_rels = [], []
        for qcity_synonym in qcity:
            for cid in city2cid[qcity_synonym]:
                all_city_pois.append(cid)
                uniqcandids2cands_city[cid] = candid2cands[cid]
                if cid in qid2rated_pool[qid]:
                    all_city_rels.append(qid2rated_pool[qid][cid])
                else:
                    all_city_rels.append(-1)
        all_city_cat_pois, all_city_cat_rels = [], []
        for cid in all_city_pois:
            if candid2cands[cid]['category'] == qcategory:
                all_city_cat_pois.append(cid)
                uniqcandids2cands_citycat[cid] = candid2cands[cid]
                if cid in qid2rated_pool[qid]:
                    all_city_cat_rels.append(qid2rated_pool[qid][cid])
                else:
                    all_city_cat_rels.append(-1)
        qid2test_pool_city[qid] = {'cands': all_city_pois, 'relevance_adju': all_city_rels}
        qid2test_pool_citycat[qid] = {'cands': all_city_cat_pois, 'relevance_adju': all_city_cat_rels}
        # Compute how good the candidate set is.
        city_match_recall = len(set.intersection(set(rel_cids), all_city_pois))/len(rel_cids)
        citycat_match_recall = len(set.intersection(set(rel_cids), all_city_cat_pois))/len(rel_cids)
        city_cands.append(len(all_city_pois))
        citycat_cands.append(len(all_city_cat_pois))
        city_recall.append(city_match_recall)
        citycat_recall.append(citycat_match_recall)
        outf.write('Query city: {:s}, country: {:s}, category: {:s}\n'.
                   format(qid2queries[qid]['city'], qid2queries[qid]['country'], qid2queries[qid]['category']))
        outf.write('City cands: {:d}; Recall: {:.2f}\n'.format(len(all_city_pois), city_match_recall))
        outf.write('City+Cat cands: {:d}; Recall: {:.2f}\n'.format(len(all_city_cat_pois), citycat_match_recall))
        outf.write(pprint.pformat(collections.Counter(rel_cats))+'\n')
        outf.write(pprint.pformat(collections.Counter(rel_cities)) + '\n')
        outf.write(pprint.pformat(collections.Counter(rel_countries)) + '\n')
        outf.write('=================\n\n')
    outf.close()
    print('City Cands: {:.2f}, Recall: {:.2f}'.format(
        statistics.mean(city_cands), statistics.mean(city_recall)))
    print('CityCat Cands: {:.2f}, Recall: {:.2f}'.format(
        statistics.mean(citycat_cands), statistics.mean(citycat_recall)))
    # Write out the cands to disk.
    with codecs.open(os.path.join(json_path, f'test-pid2all_anns-pointrec-city.json'), 'w', 'utf-8') as fp:
        json.dump(qid2test_pool_city, fp)
        print(f'Wrote: {fp.name}')
    with codecs.open(os.path.join(json_path, f'candid2all_cands-pointrec-city.json'), 'w', 'utf-8') as fp:
        json.dump(uniqcandids2cands_city, fp, indent=2)
        print(f'Wrote: {fp.name}')
    with codecs.open(os.path.join(json_path, f'test-pid2all_anns-pointrec-citycat.json'), 'w', 'utf-8') as fp:
        json.dump(qid2test_pool_citycat, fp)
        print(f'Wrote: {fp.name}')
    with codecs.open(os.path.join(json_path, f'candid2all_cands-pointrec-citycat.json'), 'w', 'utf-8') as fp:
        json.dump(uniqcandids2cands_citycat, fp, indent=2)
        print(f'Wrote: {fp.name}')


def get_pointrec_initial_cands_automatic(in_path, json_path, rank_method):
    """
    - Read the raw data and write out json files 'test-pid2anns-{dataset}.json'
        of pre-feteched candidates. The prefeteching is based on the category, subcategory,
        and the city, country of the request.
    - This function creates candidates based on the whole poi_dataset.
    """
    # Read the need.
    with codecs.open(os.path.join(in_path, 'infoneeds.json'), 'r', 'utf-8') as fp:
        qid2need = json.load(fp)
    print(f'Queries: {len(qid2need)}')
    qid2queries = collections.OrderedDict()
    for qid, need in qid2need.items():
        try:
            subcats = ', '.join(need['Sub Categories']).lower()
            meta_description = '{:s}, {:s} is known for {:s}. Specifically: {:s}'. \
                format(need['City'], need['Country'], need['Main Category'].lower(), subcats)
            subcat = need['Sub Categories']
        except TypeError:
            meta_description = '{:s}, {:s} is known for {:s}.'. \
                format(need['City'], need['Country'], need['Main Category'].lower())
            subcat = None
        qid2queries[qid] = {
            'query': meta_description,
            'city': need['City'],
            'state': need['State'],
            'country': need['Country'],
            'category': need['Main Category'].lower(),
            'subcategory': subcat
        }
    
    # Read the relevance file.
    with codecs.open(os.path.join(in_path, 'relevance/qrels.trec'), 'r', 'utf-8') as fp:
        qid2relevant_pool = collections.defaultdict(dict)
        for line in fp:
            qid, _, candid, rel = line.strip().split()
            qid2relevant_pool[qid][candid] = int(rel)
    print(f'Rated pools: {len(qid2relevant_pool)}')
    assert(len(qid2relevant_pool) == len(qid2queries))
    
    # Read the candidates.
    cid2cand_meta = dict()
    for root_dir, sub_dirs, filenames in os.walk(os.path.join(in_path, 'poi_dataset')):
        for fname in filenames:
            with codecs.open(os.path.join(root_dir, fname), 'r', 'utf-8') as fp:
                city_pois = json.load(fp)
                cid2cand_meta.update(city_pois)
    print(f'Cands read: {len(cid2cand_meta)}')
    all_candid = []
    candid2cands = collections.OrderedDict()
    missing_snippets = 0
    for candid, candd in cid2cand_meta.items():
        poi_name = candd['name']
        poi_city = candd['city']
        poi_country = candd['country_code']
        poi_category = candd['main_category'].lower()
        try:
            poi_subcategory = ', '.join(
                [s.strip() for s in candd['sub_categories'].split(',') if s.strip()]).lower()
        except KeyError:
            poi_subcategory = None
        # Get a full description for use by ranking models.
        all_snippets = candd['snippets']
        if all_snippets:
            snippets = '. '.join(['title: {:}, snippet: {:}'.
                                 format(s['title'], s['snippet'].strip()) for s in all_snippets[:5]])
        else:
            snippets = None
            missing_snippets += 1
        cand_descr = '{:s} in {:s}, {:s} is known for {:s}.'.format(poi_name, poi_city, poi_country, poi_category)
        if poi_subcategory:
            cand_descr += ' Specifically: {:s}.'.format(poi_subcategory)
        if snippets:
            cand_descr += ' {:s}.'.format(snippets)
        # Get a meta description for shortlisting candidates.
        cand_meta_descr = '{:s}, {:s} is known for {:s}.'.format(poi_city, poi_country, poi_category)
        if poi_subcategory:
            cand_meta_descr += ' Specifically: {:s}.'.format(poi_subcategory)
        all_candid.append(candid)
        candid2cands[candid] = {
            'description': cand_descr,
            'meta_description': cand_meta_descr,
            'city': poi_city,
            'country': poi_country,
            'category': poi_category,
            'subcategory': poi_subcategory
        }
    print(f'Candidates: {len(candid2cands)}')
    
    # Encode the candidates and get a nearest neighbor data structure.
    cand_strings = [d['meta_description'] for d in candid2cands.values()]
    query_text = [d['query'] for d in qid2queries.values()]
    if rank_method == 'mpnet1b':
        encoder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        cand_vecs = encoder.encode(cand_strings, batch_size=64, show_progress_bar=True)
        vec_path = os.path.join(json_path, 'poi_dataset-all_cands-mpnet1b.npy')
        np.save(vec_path, cand_vecs)
        print(f'Wrote: {vec_path}')
        with open(os.path.join(json_path, 'poi_dataset-all_candids.json'), 'w') as fp:
            json.dump(all_candid, fp)
            print(f'Wrote: {fp.name}')
        query_vecs = encoder.encode(query_text)
    elif rank_method == 'tfidf':
        encoder = skfeat.TfidfVectorizer(
            encoding='utf-8', decode_error='strict', lowercase=True, analyzer='word',
            stop_words=None, norm='l2', use_idf=True, smooth_idf=True)
        cand_vecs = encoder.fit_transform(cand_strings)
        query_vecs = encoder.transform(query_text)
    cand_index = neighbors.NearestNeighbors(n_neighbors=1000)
    cand_index.fit(cand_vecs)
    _, query_neighbors = cand_index.kneighbors(query_vecs)
    print(f'Neighbors per candidate: {query_neighbors.shape}')
    
    # Get the top candidates for each query.
    qid2recalls = {}
    qid2test_pool = dict()
    uniqcandids2cands = dict()
    num_cands = []
    outf = codecs.open(os.path.join(json_path, f'candid2all_cands-recalls-{rank_method}.txt'), 'w', 'utf-8')
    for idx, qid in enumerate(qid2queries):
        # Get the neighbors as cands for future re-ranking.
        cur_neigh_idxs = query_neighbors[idx, :].tolist()
        cur_cands = [all_candid[i] for i in cur_neigh_idxs]
        judged_rels = []
        cand_rels_binary = []
        total_rel = sum([1 if rel > 0 else 0 for rel in qid2relevant_pool[qid].values()])
        for cid in cur_cands:
            uniqcandids2cands[cid] = candid2cands[cid]
            if cid in qid2relevant_pool[qid]:
                judged_rels.append(qid2relevant_pool[qid][cid])
                if qid2relevant_pool[qid][cid] > 0:
                    cand_rels_binary.append(1)
                else:
                    cand_rels_binary.append(0)
            else:  # cid is not rated at all.
                judged_rels.append(-1)
                cand_rels_binary.append(0)
        if total_rel < 5:
            print(f'{qid} Relevant: {len(qid2relevant_pool[qid])}, {total_rel}')
        assert(len(cur_cands) == len(judged_rels) == len(cand_rels_binary))
        qid2test_pool[qid] = {'cands': cur_cands, 'relevance_adju': judged_rels}
        num_cands.append(len(cur_cands))
        # Figure out how good the ranking is.
        qrecall_100 = sum(cand_rels_binary[:100])/max(1, total_rel)
        qrecall_500 = sum(cand_rels_binary[:500])/max(1, total_rel)
        qrecall_1000 = sum(cand_rels_binary[:1000])/max(1, total_rel)
        qid2recalls[qid] = {'r100': qrecall_100, 'r500': qrecall_500, 'r1000': qrecall_1000}
        outf.write('QID: {:s}\n'.format(qid))
        outf.write('Query: {:s}\n'.format(qid2queries[qid]['query']))
        outf.write('Recall: {:.2f}, {:.2f}, {:.2f}\n'.format(qrecall_100, qrecall_500, qrecall_1000))
        cur_cands_text = '\n'.join([candid2cands[cid]['meta_description'] for cid in cur_cands[:5]])
        outf.write('{:s}\n'.format(cur_cands_text))
        outf.write('===========\n\n')
    outf.close()
    assert (len(qid2queries) == len(qid2test_pool))
    print('Pool depth; mean: {:}'.format(statistics.mean(num_cands)))
    print('Uniq cands: {:}'.format(len(uniqcandids2cands)))
    mean_recall100 = statistics.mean([qr['r100'] for qr in qid2recalls.values()])
    mean_recall500 = statistics.mean([qr['r500'] for qr in qid2recalls.values()])
    mean_recall1000 = statistics.mean([qr['r1000'] for qr in qid2recalls.values()])
    print('R@100: {:.2f}, R@500: {:.2f}, R@1000: {:.2f}'.format(mean_recall100, mean_recall500, mean_recall1000))
    # Write out the cands to disk.
    with codecs.open(os.path.join(json_path, f'test-pid2all_anns-pointrec-{rank_method}.json'), 'w', 'utf-8') as fp:
        json.dump(qid2test_pool, fp)
        print(f'Write: {fp.name}')
    out_cand_fname = os.path.join(json_path, f'candid2all_cands-pointrec-{rank_method}.json')
    with codecs.open(out_cand_fname, 'w', 'utf-8') as fp:
        json.dump(uniqcandids2cands, fp, indent=2)
        print(f'Write: {fp.name}')


def get_gpt3_predictions(in_path, out_path):
    """
    - Read the pointrec dataset.
    - Make an API call to GPT3 and 10 predictions from GPT3 model.
    - Write out the GPT3 predictions.
    """
    completion_length = 1500
    to_write = 200
    gpt3_model_name = "text-davinci-003"
    # Read the pointrec queries.
    with codecs.open(os.path.join(in_path, 'pointrec_queries.json'), 'r', 'utf-8') as fp:
        narrative_queries = json.load(fp)
        print(f'Read: {fp.name}')
        print(f'Queries: {len(narrative_queries)}')
    
    # Go over the users and write out the outputs.
    out_file_read = codecs.open(os.path.join(out_path, f'narrative_recs-{gpt3_model_name}.txt'), 'w', 'utf-8')
    labeled_prompt = yelppoi_prompts.narrative_rec_prompt_poi
    out_file_read.write(labeled_prompt)
    out_file_read.write('\n++++++++++\n\n')
    processed_queries = 0
    out_responses = {}
    for qid in sorted(narrative_queries):
        querydict = narrative_queries[qid]
        narrativeq = querydict['query']
        prompt = "{:s}\n\nUser request: {:s}\n\n10 recommendations for points of interest and their description:"\
            .format(labeled_prompt, narrativeq)
        # Make the query 3 times to the llm.
        llm_responses = []
        for i in range(3):
            try:
                response = openai.Completion.create(
                    model=gpt3_model_name,
                    prompt=prompt,
                    temperature=0.7,
                    max_tokens=completion_length,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
            except Exception:
                print('Sleeping: 10s')
                time.sleep(10)
                response = openai.Completion.create(
                    model=gpt3_model_name,
                    prompt=prompt,
                    temperature=0.7,
                    max_tokens=completion_length,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
            llm_responses.append(response)
            generated_text = response['choices'][0]['text']
            out_file_read.write(narrativeq + '\n' + generated_text + '\n\n')
        out_file_read.write('===========\n\n')
        querydict['llm_responses'] = llm_responses
        out_responses[qid] = querydict
        processed_queries += 1
        if processed_queries % 10 == 0:
            print('Users processed: {:d}'.format(processed_queries))
        if processed_queries >= to_write:
            break
    print(f'Wrote users: {processed_queries}')
    with codecs.open(os.path.join(out_path, f'narrative_recs-{gpt3_model_name}.json'), 'w', 'utf-8') as fp:
        json.dump(out_responses, fp, indent=2)
        print(f'Wrote: {fp.name}')
    out_file_read.close()


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='subcommand',
                                       help='The action to perform.')
    # Print the data for manual examination.
    print_dataset_args = subparsers.add_parser('print_dataset')
    print_dataset_args.add_argument('--in_path', required=True,
                                    help='Path to directory with raw data.')
    # Write it in json format for models to use.
    write_json_args = subparsers.add_parser('write_json')
    write_json_args.add_argument('--in_path', required=True,
                                 help='Path to directory with raw data.')
    write_json_args.add_argument('--json_path', required=True,
                                 help='Path to directory to write json outputs to.')
    # Write initial cands from whole set of candidates
    all_cands_args = subparsers.add_parser('get_all_cands')
    all_cands_args.add_argument('--in_path', required=True,
                                help='Path to directory with raw data.')
    all_cands_args.add_argument('--json_path', required=True,
                                help='Path to directory to write json outputs to.')
    all_cands_args.add_argument('--rank_method', required=True,
                                choices=['tfidf', 'mpnet1b'],
                                help='How to generate the candidates.')
    # Write gold initial cands from whole set of candidates
    allgold_cands_args = subparsers.add_parser('get_all_cands_gold')
    allgold_cands_args.add_argument('--in_path', required=True,
                                    help='Path to directory with raw data.')
    allgold_cands_args.add_argument('--json_path', required=True,
                                    help='Path to directory to write json outputs to.')
    # Generate GPT3 predictions for pointrec queries.
    get_gpt3_preds = subparsers.add_parser('get_gpt3_preds')
    get_gpt3_preds.add_argument('--in_path', required=True,
                                help='Path to directory with raw data.')
    get_gpt3_preds.add_argument('--out_path', required=True,
                                help='Path to directory to write predictions to.')
    cl_args = parser.parse_args()
    if cl_args.subcommand == 'print_dataset':
        print_dataset(in_path=cl_args.in_path)
    elif cl_args.subcommand == 'write_json':
        pointrec_to_json(in_path=cl_args.in_path, json_path=cl_args.json_path, en_snippets=False)
        print('\n')
        pointrec_to_json(in_path=cl_args.in_path, json_path=cl_args.json_path, en_snippets=True)
    elif cl_args.subcommand == 'get_all_cands':
        get_pointrec_initial_cands_automatic(in_path=cl_args.in_path, json_path=cl_args.json_path,
                                             rank_method=cl_args.rank_method)
    elif cl_args.subcommand == 'get_all_cands_gold':
        get_pointrec_initial_cands_gold(in_path=cl_args.in_path, json_path=cl_args.json_path)
    elif cl_args.subcommand == 'get_gpt3_preds':
        get_gpt3_predictions(in_path=cl_args.in_path, out_path=cl_args.out_path)


if __name__ == '__main__':
    main()
