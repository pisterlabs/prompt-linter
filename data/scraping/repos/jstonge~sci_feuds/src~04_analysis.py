import json
import re
from pathlib import Path

import numpy as np
import openai
import textwrap

import tiktoken
import pandas as pd
# import spacy
import seaborn as sns
import matplotlib.pyplot as plt

from helpers import ref_id2name_lookup

ROOT_DIR   = Path("..")
FIG_DIR    = ROOT_DIR / "figs"
OUTPUT_DIR = ROOT_DIR / 'output'
GROBID_DIR = OUTPUT_DIR / 'group_selection_grobid'
SPACY_DIR  = OUTPUT_DIR / 'spacy_group_selection_grobid'
STANCE_DIR = OUTPUT_DIR / 'stance_detection'
CCA_DIR    = OUTPUT_DIR / 'cca'

enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

pop_authors = set(pd.read_csv(OUTPUT_DIR / "list_hotshots.csv").cite_spans)
full_df = pd.read_json(OUTPUT_DIR / "My-Predictions2.json")

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

def read_article(article):
    # article = [list(GROBID_DIR.glob("*json"))]
    # article = GROBID_DIR / article
    # fname = re.sub("\.json", "", str(article).split("/")[-1])
    with open(GROBID_DIR / article) as f:
        dat = json.load(f)

    texts = [_['text'] for _ in dat['pdf_parse']['body_text']]
    cite_spans = [_['cite_spans'] for _ in dat['pdf_parse']['body_text']]
    ref_id_authors = {_['ref_id']: _['authors'] for _ in dat['pdf_parse']['bib_entries'].values()}
    
    all_texts_cite = []
    all_spans_cite = []
    for text, cite_span in zip(texts, cite_spans):
        if len(cite_span) > 0:
            all_texts_cite.append([text])
            all_spans_cite.append([ref_id2name_lookup(span['ref_id'], ref_id_authors) for span in cite_span])
        else:
            all_texts_cite.append([text])
            all_spans_cite.append([])

    return pd.DataFrame({'texts': all_texts_cite, 'cite_spans': all_spans_cite})

# special attention to Wynne-Edwards
we_lookup = {n: 'V Wynne-Edwards' for n in full_df[full_df.cite_spans.str.contains("Wynne", case=False)].cite_spans.unique()}
full_df['cite_spans'] = full_df.cite_spans.map(lambda x: we_lookup[x] if we_lookup.get(x) is not None else x)

pop_df = full_df[full_df.cite_spans.isin(pop_authors)]
print(f"Popular authors comprise {round(len(pop_df) / len(full_df)  * 100, 2)}% of the sections with authors")

def parse_reply(x):
    return re.findall("^ ?.+?(?=END|$)", x.choices[0]['text'], re.DOTALL)[0].strip()

def call_openai(x, model, max_tok=700):
    return openai.Completion.create(
        model=model, prompt=x + "\n\n###\n\n", temperature=0, max_tokens=max_tok
        )


# ------------------------------- cca analysis ------------------------------- #


step1_best_so_far = "curie:ft-personal-2023-05-01-19-28-15"

df = pd.read_parquet(CCA_DIR/"groupSel_feud_with_tag.pqt")
df = df[~df.parsed_doc.duplicated()]
df["wc"] = df.parsed_doc.str.count(" ")
df = df[df['wc'] < 500]

we_df = df[df.cite_spans.str.contains("Wynne")]
we_prompts = we_df.parsed_doc.map(lambda x: call_openai(x, step1_best_so_far))
example = we_prompts.map(parse_reply)

we_df['relevant_contexts'] = example

we_df.to_parquet("groupSel_fine_tuned_cca.parquet", index=False)


# How many sentence is a context?


def show_examples():
    tmp_df = we_df.reset_index()
    rdm_idx = tmp_df.sample(1)['index']
    print(f"given: {tmp_df.parsed_doc[rdm_idx].tolist()}\n")
    print(f"reply: {tmp_df['relevant_contexts'][rdm_idx].tolist()}")

show_examples()

# How many sentence is a context?

def plot_fig_sentece():
    d = pd.read_parquet("groupSel_fine_tuned_cca.parquet")

    nlp = spacy.load("en_core_web_trf")
    docs = list(nlp.pipe(d.relevant_contexts.tolist()))
    sent_counts = [len(list(doc.sents)) for doc in docs]

    d['sent_counts'] = sent_counts 

    f, ax = plt.subplots(1,1,figsize=(5,3))
    sns.histplot(sent_counts, ax=ax)
    ax.set_xlabel("# sentences")
    ax.set_ylabel("frequency")
    plt.title("Number of sentences\nfor relevant citation context")
    plt.tight_layout()
    plt.savefig("../../figs/multicite.pdf")
    plt.savefig("../../figs/multicite.png")


# --------------------------- cca+stance detection --------------------------- #



# best_model_so_far_3 = "curie:ft-personal-2023-05-08-21-10-03"
# best_model_so_far_5 = "ada:ft-personal-2023-05-10-17-07-55"

# best_model_so_far_id_3 = "ft-F2HCJCsDcXnUodzu37pongJf"   # balanced

# best_model_so_far_id_5 = "ft-fWXcntKe6ctUxxG2GANyneXJ" # balanced
# best_model_so_far_5 = "curie:ft-personal-2023-05-09-23-14-24"

best_model_so_far_id_5 = "ft-FrUEGnWr4iGrIPpuYKDunfJ0" # unbalanced
best_model_so_far_5 = "curie:ft-personal-2023-05-08-19-17-01"

# best_model_so_far_id_5 = "ft-CaL8qTKKA3gtbSWsVmr1vDQU" # 5/unbalanced/ada

full_df[~full_df.article.duplicated()].sort_values('citations', ascending=False)

tidy_article = read_article("hamilton_genetical_1964.json").explode('texts')
target_article = full_df.query("article == 'puhalskii_large-population_2017'")
# text = puhalskii.abstract[i]
# puhalskii[puhalskii.sid == 1]
# stance_pred = puhalskii.stance[i]
text = full_df.loc[full_df.stance.argmin(), :].abstract

target_article_dedup = target_article[~target_article.sid.duplicated()]

res = openai.Completion.create(
    model=best_model_so_far_5, 
    prompt=text + "\n\n###\n\n", 
    max_tokens=1, temperature=0, logprobs=2
)


## READING ALL CITATIONS
from tqdm import tqdm


all_res = []
for text in tqdm(target_article_dedup.abstract[5:]):
    res = openai.Completion.create(model=best_model_so_far_5, prompt=text + "\n\n###\n\n", max_tokens=1, temperature=0, logprobs=2)
    all_res.append(res)

len(all_res)

all_res_parsed = [parse_reply(r) for r in all_res]


# we_prompts = we_df.abstract.map(lambda x: call_openai(x, best_model_so_far))



reply = parse_reply(res)
# res.choices[0]['logprobs']['top_logprobs'][0]
print(f"**Prediction curie:** {reply}\n\n**Prediction scibert:** {stance_pred}\n\n**text:** {text}")



# Protypical example of what we want.

most_neg_articles = full_df.groupby("article")['stance'].mean().sort_values().head(30).index.tolist()
most_pos_articles = full_df.groupby("article")['stance'].mean().sort_values().tail(30).index.tolist()

full_df.loc[full_df.article.isin(most_pos_articles)].value_counts(["did", "sid"]).reset_index(name="n").value_counts("did")

full_df[full_df.article == 'sober_principle_1981']
full_df[full_df.did == 85]

target_article = full_df.query("article == 'puhalskii_large-population_2017'")
target_article_dedup = target_article[~target_article.sid.duplicated()]

target_article2 = full_df.query("article == 'durham_adaptive_1976'")
target_article2_dedup = target_article2[~target_article2.sid.duplicated()]

target_article3 = full_df.query("article == 'alcock_myth_1999'")
target_article3_dedup = target_article3[~target_article3.sid.duplicated()]

target_article4 = full_df.query("article == 'brune_schizophreniaevolutionary_2004'")
target_article4_dedup = target_article4[~target_article4.sid.duplicated()]

target_articl5 = full_df.query("article == 'trescases_triangular_2016'")
target_article5_dedup = target_articl5[~target_articl5.sid.duplicated()]

target_article_we = full_df[full_df.cite_spans.str.contains("Wynne")]
target_article_we = target_article_we[target_article_we.year != 'Kurt et al. - 2021 - Two-dimensionally stable self-organization arises ']
target_article_we['year'] = target_article_we.year.map(lambda x: x.split("-")[0])
target_article_we['year'] = pd.to_datetime(target_article_we.year, format="%Y")
target_article_we = target_article_we[~target_article_we[['year', 'article', 'did', 'sid']].duplicated()]
target_article_we = target_article_we.sort_values(["year", "article", "did", "sid"])


current_idx = 0

all_res13 = []
for text in tqdm(target_article_we.abstract[current_idx:]):

    prompt = f"""
    Identify the following items from the review text: 
        - Stance towards the unique author  that is delimited by the  \
        <cite> tag </cite> (ranging  from -1 to 1 where -1=very \
        negative stance; 0=neutral stance; 1=very positive stance)
        - Identify key words that explain the stance.
        - Target citation name is the author within the tag

    Format your response as a JSON object with "Stance", \
    "key words" and "target"  as the keys.

    text: '''{text}'''"""

    res = get_completion(prompt)
    all_res13.append(res)

print(res)

current_idx = len(all_res) + len(all_res2) + len(all_res3) + len(all_res4) + len(all_res5) + len(all_res6) + len(all_res7) + len(all_res8) + len(all_res9) + len(all_res10) + len(all_res11) + len(all_res12) 

# len(all_res)+len(all_res2)+len(all_res3)

true_all_res = all_res + all_res2 + all_res3 + all_res4 + all_res5 + all_res6 + all_res7 + all_res8 + all_res9 + all_res10 + all_res11 + all_res12 + all_res13
assert len(true_all_res) == len(target_article_we)


true_all_res[46] = true_all_res[46][:138]

df_gold = pd.DataFrame([json.loads(res) for res in true_all_res])


df_gold.to_pickle(STANCE_DIR / "gpt3-5-turbo" / "wynned_edwards_all_cite_spans.pkl")

done_gold = list(STANCE_DIR.joinpath("gpt3-5-turbo").glob("*pkl"))

def plot_diff(df1, df2):
    df1 = pd.concat([df1 , df2.reset_index(drop=True)], axis=1)

    f, ax = plt.subplots(1,1)
    sns.lineplot(x="sid", y="Stance", label="gpt3.5", data=df1, ax=ax)
    sns.lineplot(x="sid", y="stance", label="scibert" ,data=df1, ax=ax)

df_gold4=pd.read_pickle(done_gold[4])
df_gold0=pd.read_pickle(done_gold[0])
df_gold0
plot_diff(df_gold0, target_article_dedup)


text_analysis_gold = pd.concat([pd.read_pickle(f) for f in done_gold], axis=0)

text_analysis_gold_long = text_analysis_gold.explode("key words")

text_analysis_gold_long.query("Stance > 0.5").value_counts("key words").head(30)



def wrap_labels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_yticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                      break_long_words=break_long_words))
    ax.set_yticklabels(labels, rotation=0)