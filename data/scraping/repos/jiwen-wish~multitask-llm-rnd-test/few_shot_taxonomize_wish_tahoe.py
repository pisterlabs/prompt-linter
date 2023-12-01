# %%
import os
import openai
import time 

openai.api_key = os.environ['OPENAI_API_KEY']
assert os.environ['OPENAI_API_KEY'] != ""

# TODO: change this for different split
mode = 'Train'

current_folder = os.path.dirname(__file__)
df_labelled_path = os.path.join(
		current_folder,
		'..',
		'..',
		'data',
		'wish_products',
		'wish-mturk-labelled-09202022-clean-joinedlance.json'
)

df_wish_tahoe_path = os.path.join(
		current_folder,
		'..',
		'..',
		'data',
		'wish_products_truetag_tahoe',
		f'Wish_Truetag_Tahoe_Meta_{mode}.json'
)

df_wish_tahoe_out_path = os.path.join(
		current_folder,
		'..',
		'..',
		'data',
		'wish_products_truetag_tahoe',
		f'Wish_Truetag_Tahoe_Meta_{mode}_OpenaiInferred.json'
)

# %%
import pandas as pd

# %%
df = pd.read_json(df_labelled_path, lines=True)

# %%

import json
from tqdm import trange

# OPTION: change ... in rand() <= ... for different sample ratio
# OPTION: change cat to tac or vice versa to go from bottom to top of file

def iter_pse(pse):
    for l in pse:
        yield l

with os.popen("cat " + df_wish_tahoe_path + \
		" | awk 'BEGIN {srand()} !/^$/ { if (rand() <= .002) print $0}'") as pse:
	while True:
		with open(df_wish_tahoe_out_path, 'a', buffering=1) as fout:
			for _ in trange(10000):
				line = next(iter_pse(pse), None)
				if line is None:
					print('file stream exhausted')
					raise Exception('filestream exhausted')
				try:
					rec = json.loads(line)
					title = rec['title'].strip()
					response = openai.Completion.create(
						model="text-davinci-002",
						prompt='\n'.join(df.sample(30)['text'].tolist()) + '\n' + title + " ->",
						temperature=0,
						max_tokens=256,
						top_p=1,
						frequency_penalty=0,
						presence_penalty=0,
						stop=["\n"]
					)

					tax = response['choices'][0]['text'].strip()
					rec['openai_category'] = tax[1:-1].split('][')
					fout.write(json.dumps(rec))
					fout.write('\n')
				except Exception as e:
					print(e)
					time.sleep(1)
		os.system(f"dvc add {df_wish_tahoe_out_path}")
		time.sleep(10)
		print(f'{df_wish_tahoe_out_path} added to dvc')
		os.system(f"dvc repro process_wish_truetag_tahoe_openai_inferred_new_tax")
		time.sleep(10)
		print(f'process_wish_truetag_tahoe_openai_inferred_new_tax stage ran')
		