#%%
import pandas as pd
import dvc.api
from tqdm import tqdm
from openai.embeddings_utils import get_embedding
import json

df = pd.concat([
    pd.read_csv(dvc.api.get_url(
        'modelling/notebooks/query_attr_extract_appen_label/appen_query_attribution_batch1.csv',
        repo='git@github.com:ContextLogic/multitask-llm-rnd.git'
    )),
    pd.read_csv(dvc.api.get_url(
        'modelling/notebooks/query_attr_extract_appen_label/appen_query_attribution_batch2.csv',
        repo='git@github.com:ContextLogic/multitask-llm-rnd.git'
    )),
    pd.read_csv(dvc.api.get_url(
        'modelling/notebooks/query_attr_extract_appen_label/appen_query_attribution_batch3.csv',
        repo='git@github.com:ContextLogic/multitask-llm-rnd.git'
    )),
    pd.read_csv(dvc.api.get_url(
        'modelling/notebooks/query_attr_extract_appen_label/appen_query_attribution_batch4.csv',
        repo='git@github.com:ContextLogic/multitask-llm-rnd.git'
    ))
])

#%%
df_exist = pd.concat([
    pd.read_json('/workspaces/multitask-llm-rnd/datasets/data/query_attr_extract_label/processed2/appen_020323_delivered_022123_validated_query_attr_train_oaiemb.json', lines=True),
    pd.read_json('/workspaces/multitask-llm-rnd/datasets/data/query_attr_extract_label/processed2/appen_020323_delivered_022123_validated_query_attr_val_oaiemb.json', lines=True),
    pd.read_json('/workspaces/multitask-llm-rnd/datasets/data/query_attr_extract_label/processed2/appen_020323_delivered_022123_validated_query_attr_test_oaiemb.json', lines=True)
])[['query']]

#%%
df_merge = df.drop_duplicates('query').merge(df_exist.drop_duplicates('query'), on='query', 
    how='left', indicator=True)
df_left = df_merge[df_merge['_merge'] == 'left_only']
# %%
engine = 'text-embedding-ada-002'
with open('appen_tolabel_query_attr_leftover.json', 'a') as f:
    for q in tqdm(df_left['query'].to_list()):
        try:
            i = {
                'openai_embedding': get_embedding(q, engine=engine),
                'query': q 
            }
            f.write(json.dumps(i) + '\n')
        except Exception as e:
            print(f"query: {q} ; exception: {e}")