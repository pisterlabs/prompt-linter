import re
from pathlib import Path

import openai
import pandas as pd

ROOT_DIR = Path("..")
OUTPUT_DIR = ROOT_DIR / 'output'
GROBID_DIR = OUTPUT_DIR / 'group_selection_grobid'
SPACY_DIR = OUTPUT_DIR / 'spacy_group_selection_grobid'
CCA_DIR = OUTPUT_DIR / 'cca'

openai.api_key = open("myapikey.txt", "r").read().strip("\n")


def prep_data():

    # We use citation window = 5
    df_5 = pd.concat([pd.read_json(CCA_DIR / f"{x}_5.json") for x in ['train', 'test', 'dev']], axis=0)
    df_gold = pd.concat([pd.read_json(CCA_DIR / f"{x}_gold.json") for x in ['train', 'test', 'dev']], axis=0)
    df_combined = df_5.merge(df_gold, on='id', how='left', suffixes=['_5', '_gold'])

    # at first we just sampled at random
    # df_prompt_completion_1 = pd.read_csv(f"../output/.cache_fine_tuned/first_iteration_cca_2023-05-01.csv", names=['index', 'prompt', 'completion'], skiprows=1, index_col='index')
    # only the longest sentences b/c our model struggled with it
    # df_prompt_completion_3 = pd.read_csv(f"../output/.cache_fine_tuned/third_iteration_cca_2023-05-01.csv", names=['index', 'prompt', 'completion'], skiprows=1, index_col='index')
    # df_prompt_completion = pd.concat([df_prompt_completion_1, df_prompt_completion_3], axis=0)

    # done_idx = df_combined.loc[df_prompt_completion.index, :].id
    # df_combined = df_combined[df_combined.id.isin(done_idx)]

    prompt = df_combined.x_5.map(lambda x: str(x) + "\n\n###\n\n")
    completion = df_combined.x_gold.map(lambda x: " " + str(x) + "END")

    df_prompt_completion = pd.DataFrame(zip(prompt, completion), columns=['prompt', 'completion'])

    df_prompt_completion.to_json("../output/context_detection_2023-05-05.jsonl", lines=True, orient='records')

    # saving what we did so far. To update
    # df_prompt_completion.to_csv(f"../output/.cache_fine_tuned/third_iteration_cca_{str(date.today())}.csv")


# fine tuning
#! openai tools fine_tunes.prepare_data -f context_detection_2023-05-08.jsonl
#! openai -k "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" fine_tunes.create -m curie -t "context_detection_2023-05-05_prepared.jsonl"


# Validating the model by comparing with multicite annotated data
best_so_far = "curie:ft-personal-2023-05-01-19-28-15"
df_gold = pd.concat([pd.read_json(CCA_DIR / f"{x}_gold.json") for x in ['train', 'test', 'dev']], axis=0)


# best_so_far = "curie:ft-personal-2023-05-01-23-05-22"

# def print_example(df, mod, max_wc=100, min_wc=0):
#     df = df[(df.x_gold.str.count(" ") >= min_wc) & (df.x_gold.str.count(" ") <= max_wc)]
#     # rdm_idx = df.sample(1).index[0]
#     rdm_idx = 6901
    
#     test1 = " However, convolutional models must be significantly deeper to retrieve the same temporal receptive field [23] . Recently, the mechanism of self-attention<cite> [22,</cite> 24] was proposed, which uses the whole sequence at once to model feature interactions that are arbitrarily distant in time. Its use in both encoder-decoder and feedforward contexts has led to faster training and state-of-the-art results in translation (via the Transformer<cite> [22]</cite> ), sentiment analysis [25] , and other tasks. These successes have motivated preliminary work in self-attention for ASR. Time-restricted self-attention was used as a drop-in replacement for individual layers in the state-of-theart lattice-free MMI model [26] , an HMM-NN system."
#     target_x = "Its use in both encoder-decoder and feedforward contexts has led to faster training and state-of-the-art results in translation (via the Transformer<cite> [22]</cite> ), sentiment analysis [25] , and other tasks."
#     # test1 = df_todo[rdm_idx]
#     # target_x = df_todo[rdm_idx]

#     res = openai.Completion.create(model=mod, prompt=test1 + "\n\n###\n\n", temperature=0, max_tokens=500)
#     reply_content=re.findall("^ ?.+?(?=END|$)", res.choices[0]['text'], re.DOTALL)[0].strip()
#     print(f"rdm_idx: {rdm_idx}\n\ngiven: {test1}\n\ntarget: {target_x}\n\nreply: {reply_content}")

# print_example(df_todo, best_so_far, max_wc=40)





# Trying to compare Wynne-Edwards 60s vs today using allotax (hint: not good)


# def clean_toks(x):
#     return [w.lemma_ for w in x if w.is_stop == False and w.pos_ not in ['PUNCT'] 
#             and w.text not in ['<', '>', '/', '<?/?cite>?', 'Wynne', 'Edwards', '1962'] and len(w.text) > 1]

# nlp = spacy.load("en_core_web_trf")

# we_df['relevant_contexts'] = we_df.relevant_contexts.str.replace("(<cite>|</cite>)", " ", regex=True)
# we_df['toks'] = list(nlp.pipe(we_df.relevant_contexts))


# we_df['types'] = we_df['toks'].map(clean_toks)

# def print_example_cleaning():
#     print(we_df['relevant_contexts'][0], end="\n\n")
#     print([w.text for w in we_df["toks"][0]], end="\n\n")
#     print([w for w in we_df["clean_toks"][0]])

# print_example_cleaning()


# we_df_early = we_df[we_df.year < "1995-01-01"]
# we_df_today = we_df[we_df.year > "1995-01-01"]

# we_df_early = we_df_early.explode("types").value_counts('types').reset_index(name='counts') 
# we_df_today = we_df_today.explode("types").value_counts('types').reset_index(name='counts') 

# we_df_early['probs'] = we_df_early.counts / we_df_early.counts.sum()
# we_df_today['probs'] = we_df_today.counts / we_df_today.counts.sum()

# we_df_early['totalunique'] = len(we_df_early.types.unique())
# we_df_today['totalunique'] = len(we_df_today.types.unique())

# we_df_early.to_csv("we_df_early.csv", index=False)
# we_df_today.to_csv("we_df_today.csv", index=False)








