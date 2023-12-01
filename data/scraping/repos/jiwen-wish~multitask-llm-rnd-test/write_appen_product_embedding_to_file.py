#%%
import pandas as pd
import dvc.api
from tqdm import tqdm
from openai.embeddings_utils import get_embedding
import json

df = pd.concat([
    pd.read_csv('../../data/wish_attr_extract_label/appen/input_batch_processed/appen_product_attribution_batch1.csv'),
    pd.read_csv('../../data/wish_attr_extract_label/appen/input_batch_processed/appen_product_attribution_batch2.csv'),
    pd.read_csv('../../data/wish_attr_extract_label/appen/input_batch_processed/appen_product_attribution_batch3.csv'),
])

df['text'] = 'Product Title: ' + df['title'].astype(str) + ' ; ' + 'Product Description: ' + df['product_description'].astype(str)
df['text'] = df['text'].apply(lambda x: x.replace('\n', ' '))
#%%
engine = 'text-embedding-ada-002'
with open('appen_tolabel_product_attr_leftover.json', 'a') as f:
    for d in tqdm(df.to_dict('records')):
        try:
            i = {
                'openai_embedding': get_embedding(d['text'], engine=engine),
                'text': d['text'],
                'product_id': d['product_id'],
                'label_ordering': d['label_ordering']
            }
            f.write(json.dumps(i) + '\n')
        except Exception as e:
            print(f"text: {d['text']} ; exception: {e}")