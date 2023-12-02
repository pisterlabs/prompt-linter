
from glob import glob
import pandas as pd
import tiktoken
from pathlib import Path

import openai
from tqdm import tqdm
from dotenv import load_dotenv
assert load_dotenv(), "Failed to load .env file"

pd.options.display.max_rows = 999
pd.options.display.max_columns = 200
    
# Estimate embedding cost
EMBEDDING_COST_PER_TOKEN = 0.0001 / 1000

# Will stdout a warning if any articles 
# are approximately longer than this many tokens
MAX_TOKEN_TO_EMBED = 8191

# The model id for the text embedding model
MODEL_ID = "text-embedding-ada-002"


#####################
#####################
#####################

if __name__ == "__main__":


    print("Reading in data...")

    # read in all the parquet files
    nyt_files = glob("data/NYT/*/*.parquet")
    articles = pd.concat([pd.read_parquet(f) for f in nyt_files])
    articles.reset_index(drop=True, inplace=True)

    symbols_path = Path("data/symbols.parquet")
    symbols = pd.read_parquet(symbols_path)

    # read in the keyword article mapper from previous script
    keyword_article_mapper_path = Path("data/keyword_article_mapper.parquet")
    keyword_article_mapper = pd.read_parquet(keyword_article_mapper_path)

    # read in the manual review
    match_review_dtypes = {
        "default_match_override": pd.BooleanDtype(),
        "default_match": pd.BooleanDtype(),
        "keyword_norm_id": pd.Int64Dtype(),
        "symbol": pd.StringDtype(),
    }

    match_review_path = Path("data/match_review_complete.csv")
    if not match_review_path.exists():
        # if no manual review exists, just use the automatic matches
        match_review_path = Path("data/match_review.csv")

    review = pd.read_csv(match_review_path, dtype=match_review_dtypes)

    print("Parsing matches...")

    # fill in the default match override with the manual review
    is_match = review['default_match_override'].fillna(review['default_match'])
    review = review[is_match]
    keyword_symbol_mapper = review.set_index('keyword_norm_id')['symbol']

    article_symbol_mapper = keyword_article_mapper.join(keyword_symbol_mapper, on='keyword_norm_id', how='inner')
    article_symbol_mapper = article_symbol_mapper['symbol']

    # get list of article ids that have been matched
    # so we know which articles to embed
    relevant_article_ids = article_symbol_mapper.index

    # filter for only the articles that contain a keyword
    # which was matched to a symbol
    relevant_articles = articles[articles['_id'].isin(relevant_article_ids)].reset_index(drop=True)


    ###################
    # PLOTS

    print("Creating Plots...")

    weekly_counts = relevant_articles.groupby(pd.Grouper(key='pub_date', freq='W'))['_id'].count()
    ax = weekly_counts.plot(figsize=(15, 5), title='Number of Relevant Articles per Week')
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Articles')
    ax.get_figure().savefig('figures/weekly_article_counts.png')

    ###################

    mentions = relevant_articles[["pub_date", "_id"]].merge(
        article_symbol_mapper, left_on="_id", right_index=True, how="left"
    )
    mentions_filepath = Path("data/mentions.parquet")
    mentions.drop(columns='pub_date').to_parquet(mentions_filepath)

    mentions_per_article = mentions['_id'].value_counts().value_counts().rename('mention_count')
    mentions_per_article = mentions_per_article.loc[:4]
    ax = mentions_per_article.plot(kind='bar', figsize=(10, 5), title='Number of Articles per Mention Count')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_xlabel('Number of Companies Mentioned in Article')
    ax.set_ylabel('Number of Articles');
    ax.get_figure().savefig('figures/mentions_per_article.png')

    ###################

    company_mentions = mentions.groupby("symbol").agg(**{
        "Num. Mentions":("_id", "count"),
        "Num. Articles":("_id", "nunique")
    })

    top_n = 20
    company_mentions = company_mentions.sort_values("Num. Mentions", ascending=False)
    ax = company_mentions.iloc[:top_n].plot(kind='bar', figsize=(10, 5), title=f'Number of Mentions per Company (Top {top_n})')
    ax.set_xlabel('Company Symbol')

    ax.get_figure().savefig('figures/mentions_per_company.png')

    ###################

    # join in GICS sector
    mentions = mentions.merge(symbols[['symbol', 'gics_sector']], left_on='symbol', right_on='symbol', how='left')
    # group by month and sector
    mentions_by_month = mentions.groupby([pd.Grouper(key='pub_date', freq='M'), 'gics_sector']).size().rename('Num. Mentions').reset_index()
    # plot
    ax = mentions_by_month.pivot(index='pub_date', columns='gics_sector', values='Num. Mentions').plot(figsize=(15, 5), title='Number of Mentions per Month by Sector')
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Mentions')
    ax.get_figure().savefig('figures/mentions_by_month_sector.png')

    ###################
    # EMBEDDING COST

    print("Estimating Embedding Cost...")

    # pull out the text that will be embedded
    relevant_articles["article_text"] = relevant_articles["headline.main"] + " " + relevant_articles["lead_paragraph"] 
    lead_paragraph_not_abstract = relevant_articles["lead_paragraph"] != relevant_articles["abstract"]
    relevant_articles.loc[lead_paragraph_not_abstract, "article_text"] += (" " + relevant_articles["abstract"])

    # Estimate embedding cost
    encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
    relevant_articles["n_token"] = relevant_articles["article_text"].apply(lambda x: len(encoding.encode(x)))

    if relevant_articles["n_token"].max() > MAX_TOKEN_TO_EMBED:
        print("Warning: some articles are too long to embed")

    n_tokens = relevant_articles["n_token"].sum()
    cost_in_dollars = n_tokens * EMBEDDING_COST_PER_TOKEN
    print(
        f"It will cost approximately ${cost_in_dollars:.2f} to embed all the articles, which is {n_tokens:,} tokens"
    )

    print("Starting Article Embeddings...")

    embeddings_filepath = Path('data/nyt_embeddings.parquet')
    if embeddings_filepath.exists():
        print("Embeddings already exist. Reading in to avoid unnecessary cost...")
        embedding_df = pd.read_parquet(embeddings_filepath)
        relevant_articles = relevant_articles[~relevant_articles['_id'].isin(embedding_df.index)]
    else:
        print("Embeddings do not exist. Creating new dataframe...")
        embedding_df = pd.DataFrame()

    # no need to embed if there are no articles to embed
    if len(relevant_articles) > 0:

        batch_size = 100
        save_every_n_batches = 50
        batch_ids = relevant_articles.index // batch_size
        batches = relevant_articles.groupby(batch_ids)
        progress_bar = tqdm(batches, total=len(batches))

        for batch_id, batch in progress_bar:
            
            # skip if already embedded
            input = batch['article_text'].tolist()
            response = openai.Embedding.create(input=input, model=MODEL_ID)
            embedding_lists = [d['embedding'] for d in response['data']]
            embeddings_batch = pd.DataFrame(embedding_lists, index=batch['_id'])
            embedding_df = pd.concat([embedding_df, embeddings_batch])

            # save every few batches
            if batch_id % save_every_n_batches == 0:
                embedding_df.to_parquet(embeddings_filepath)
                progress_bar.set_description(f"Saved batch {batch_id}")

        # save the remaining embeddings
        print("Done embedding all articles. Saving...")
        embedding_df.to_parquet(embeddings_filepath)
        print(f"Saved embeddings for batch {batch_id}")
    else:
        print("No articles to embed.")

    print("Done.")