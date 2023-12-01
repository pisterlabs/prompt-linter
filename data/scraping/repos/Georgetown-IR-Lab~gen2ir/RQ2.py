import pandas as pd
import pyterrier as pt

pt.init()
from time import sleep
import json
from tqdm import tqdm
import ir_datasets
import openai
import os
from openai.error import RateLimitError
import pyterrier_dr
import pyterrier_pisa
import heapq
import random

from pyterrier_t5 import MonoT5ReRanker

monoT5 = MonoT5ReRanker()  # loads castorini/monot5-base-msmarco by default

#DEPTH = 2
DOC_DEPTH = 10
no_of_mutations_per_iteration = 8

savedf = pd.DataFrame(columns=['Query', 'Result_O', 'MonoT5_O', 'Result_M', 'MonoT5_M', 'Result_Gen', 'MonoT5_Gen', 'Judge'])

openai.api_key = ''

bm25 = pyterrier_pisa.PisaIndex.from_dataset('msmarco_passage').bm25(num_results=1000)
electra = pyterrier_dr.ElectraScorer(verbose=False)

pipeline = bm25 >> pt.text.get_text(pt.get_dataset('irds:msmarco-passage'), 'text') >> electra

model, prefix = 'text-davinci-edit-001', 'davinci'


def query_rewrite(docid, passage, query, count=1):
    while True:
        try:
            result = openai.Edit.create(
                engine='text-davinci-edit-001',
                input=passage,
                instruction='Re-write the passage to better answer the question: ' + query,
                api_key=os.getenv('OPENAI'),
                temperature=0.7,
                top_p=1,
                n=count,
            )
            break
        except openai.error.RateLimitError:
            print('RATE LIMIT, sleeping for 10 seconds')
            sleep(10)
    for idx, g in enumerate(result['choices']):
        if 'text' not in g.keys():
            text = ''
            print('oops')
        else:
            text = g['text'].strip().replace('\n', ' ').strip()
        yield {'docno': f'{docid}-qr{idx}', 'text': text}


def rewrite(docid, passage, count=1):
    while True:
        try:
            result = openai.Edit.create(
                engine='text-davinci-edit-001',
                input=passage,
                instruction='Re-write the passage',
                api_key=os.getenv('OPENAI'),
                temperature=0.7,
                top_p=1,
                n=count,
            )
            break
        except openai.error.RateLimitError:
            print('RATE LIMIT, sleeping for 10 seconds')
            sleep(10)
    for idx, g in enumerate(result['choices']):
        if 'text' not in g.keys():
            text = ''
            print('oops')
        else:
            text = g['text'].strip().replace('\n', ' ').strip()
        yield {'docno': f'{docid}-r{idx}', 'text': text}


def combine(docno1, docno2, passage1, passage2, query, count=1):
    while True:
        try:
            result = openai.Edit.create(
                engine='text-davinci-edit-001',
                input=f'Passage1: {passage1}\n\nPassge2: {passage2}\n\nAnswer:',
                instruction='Combine ideas from both Passage1 and Passage2 to answer the question: ' + query,
                api_key=os.getenv('OPENAI'),
                temperature=0.7,
                top_p=1,
                n=count,
            )
            break
        except openai.error.RateLimitError:
            print('RATE LIMIT, sleeping for 10 seconds')
            sleep(10)
    for idx, g in enumerate(result['choices']):
        text = ''
        if 'text' in g.keys():
            if '\n\nAnswer:' in g['text']:
                text = g['text'].split('\n\nAnswer:')[1]
        else:
            print('oops')
        yield {'docno': f'({docno1}+{docno2})-{idx}', 'text': text.replace('\n', ' ').strip()}


# for ds, dsid in [('dl19', 'msmarco-passage/trec-dl-2019/judged'), ('dl20', 'msmarco-passage/trec-dl-2020/judged')]:
for ds, dsid in [('dev', 'msmarco-passage/dev/small')]:
    query_count = 0
    for i, query in enumerate(tqdm(ir_datasets.load(dsid).queries)):
        query_count += 1
        print(query)
        res = pipeline.search(query.text)
        # print(res['text'].iloc[2])
        original = res['text'].iloc[0]
        print('original')
        print(original)
#################################################################
        # Add re-ranking retrieval results to heap
        heapo = [(float('-inf'), '', '')]
        for item in res.itertuples(index=False):
            heapo.append((item.score, item.docno, item.text))
        heapo = sorted(heapo)
        res = []

        for iteration in range(7): #constant budget determined from percentile
            # Mutations
            for n in range(no_of_mutations_per_iteration): #constant budget determined from percentile
                case = random.random()
                try:
                    if case <= 0.33:
                        docid = int(random.random() * 100) % DOC_DEPTH + 1
                        res.extend(rewrite(heapo[-1 * docid][1], heapo[-1 * docid][2]))
                    elif case <= 0.66:
                        docid = int(random.random() * 100) % DOC_DEPTH + 1
                        res.extend(query_rewrite(heapo[-1 * docid][1], heapo[-1 * docid][2], query.text))
                    else:
                        docid1 = int(random.random() * 100) % DOC_DEPTH + 1
                        docid2 = int(random.random() * 100) % DOC_DEPTH + 1
                        if docid1 == docid2:
                            docid2 += 1
                        res.extend(combine(heapo[-1 * docid1][1], heapo[-1 * docid2][1], heapo[-1 * docid1][2],
                                           heapo[-1 * docid2][2], query.text))
                except:
                    print('err_er')
                    continue
            # Evaluate new documents
            # print(res)
        res = pd.DataFrame({'qid': query.query_id, 'query': query.text, 'docno': [x['docno'] for x in res],
                            'text': [x['text'] for x in res]})
        res = electra(res)

            # Add new documents to heap
        for item in res.itertuples(index=False):
            heapo.append((item.score, item.docno, item.text))
        heapo = sorted(heapo)


        print('mutated')
        print(heapo[-1][2])
        # break

################################################################
        # Add re-ranking retrieval results to heap
        heap = [(float('-inf'), '', '')]
        for item in res.itertuples(index=False):
            heap.append((item.score, item.docno, item.text))
        heap = sorted(heap)

        for iteration in range(7):#constant budget determined from percentile
            # Mutations
            res = []
            for n in range(no_of_mutations_per_iteration):#constant budget determined from percentile
                case = random.random()
                try:
                    if case <= 0.33:
                        docid = int(random.random() * 100) % DOC_DEPTH + 1
                        res.extend(rewrite(heap[-1 * docid][1], heap[-1 * docid][2]))
                    elif case <= 0.66:
                        docid = int(random.random() * 100) % DOC_DEPTH + 1
                        res.extend(query_rewrite(heap[-1 * docid][1], heap[-1 * docid][2], query.text))
                    else:
                        docid1 = int(random.random() * 100) % DOC_DEPTH + 1
                        docid2 = int(random.random() * 100) % DOC_DEPTH + 1
                        if docid1 == docid2:
                            docid2 += 1
                        res.extend(combine(heap[-1 * docid1][1], heap[-1 * docid2][1], heap[-1 * docid1][2],
                                           heap[-1 * docid2][2], query.text))
                except:
                    print('err_er')
                    continue
            # Evaluate new documents
            # print(res)
            res = pd.DataFrame({'qid': query.query_id, 'query': query.text, 'docno': [x['docno'] for x in res],
                                'text': [x['text'] for x in res]})
            res = electra(res)

            # Add new documents to heap
            for item in res.itertuples(index=False):
                heap.append((item.score, item.docno, item.text))
            heap = sorted(heap)


        print('final')
        print(heap[-1][2])
        # break

        inp0 = pd.DataFrame([['q1', query, 'd1', original]], columns=['qid', 'query', 'docno', 'text'])
        mono_score_o = monoT5.transform(inp0).loc[0].at["score"]
        inp1 = pd.DataFrame([['q1', query, 'd1', heapo[-1][2]]], columns=['qid', 'query', 'docno', 'text'])
        mono_score_m = monoT5.transform(inp1).loc[0].at["score"]
        inp2 = pd.DataFrame([['q1', query, 'd1', heap[-1][2]]], columns=['qid', 'query', 'docno', 'text'])
        mono_score_gen = monoT5.transform(inp2).loc[0].at["score"]
        if mono_score_gen - mono_score_m > 0:
            judgeval = 1
        elif mono_score_gen - mono_score_m == 0:
            judgeval = 0
        else:
            judgeval = -1  
        savedf = savedf.append(
            {'Query': query, 'Result_O': original, 'MonoT5_O': mono_score_o, 'Result_M': heapo[-1][2], 'MonoT5_M': mono_score_m, 'Result_Gen': heap[-1][2],
             'MonoT5_Gen': mono_score_gen, 'Judge': judgeval}, ignore_index=True)

        if query_count == 100:
            print(savedf)
            break
savedf.to_csv('outputrq2.csv', index=False)
