from data import train, test, user_id_list, ratings_df, movies_df
from random import sample
import os
import openai
import re
import pandas as pd
from retry import retry
from typing import Callable, Any


openai.api_key = os.getenv("OPENAI_API_KEY")

SHOT_RATING_THRESHOLD = 5
PREDICTED_RATING_THRESHOLD = 8

SAMPLE_COUNT = 500

RECOMMENDING_PROMPT = """
You are a movie recommender system now.

Here is the watching history of a user: {}.

The user has given high ratings to the provided movies. Based on this history, please predict the user’s rating for the following item: {} (1 being lowest and 10 being highest)

You should wrap the rating number with the ` (Backtick) so that the program can parse it. You MUST NOT include any other characters in your response except the rating. You MUST represent ratings not in RANGE but a SINGLE NUMBER.
"""

# NOTE: Sampling a few users due to the cost of ChatGPT API
sampled_users_id_list = sample(user_id_list, 1)


# NOTE: Apply retrying feature for rate-limit
@retry(tries=3, delay=10)
def make_response_to_gpt(prompt):
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]
    )


@retry(tries=4)
def get_rating_from_gpt(create_response: Callable[..., Any]) -> int:
    response = create_response()
    message = response.choices[0].message.content  # type: ignore

    match = re.search(r"`(\d+)`", message)

    if match:
        rating_string = match.group(1)
        return int(rating_string)
    else:
        raise


def create_confusion_matrix(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    empty_df = pd.DataFrame(columns=df.columns)

    tp_df_list = []
    tn_df_list = []
    fp_df_list = []
    fn_df_list = []

    for _, row in df.iterrows():
        # NOTE: TP
        if (row["predicted_ratings"] >= PREDICTED_RATING_THRESHOLD) & row[
            "exists_in_watched"
        ]:
            tp_df_list.append(row.to_dict())
            continue
        # NOTE: FP
        if (row["predicted_ratings"] >= PREDICTED_RATING_THRESHOLD) & ~row[
            "exists_in_watched"
        ]:
            fp_df_list.append(row.to_dict())
            continue
        # NOTE: TN
        if (row["predicted_ratings"] < PREDICTED_RATING_THRESHOLD) & ~row[
            "exists_in_watched"
        ]:
            tn_df_list.append(row.to_dict())
            continue
        # NOTE: FN
        if (row["predicted_ratings"] < PREDICTED_RATING_THRESHOLD) & row[
            "exists_in_watched"
        ]:
            fn_df_list.append(row.to_dict())
            continue

    tp_df = pd.DataFrame(tp_df_list) if len(tp_df_list) != 0 else empty_df
    fp_df = pd.DataFrame(fp_df_list) if len(fp_df_list) != 0 else empty_df
    tn_df = pd.DataFrame(tn_df_list) if len(tn_df_list) != 0 else empty_df
    fn_df = pd.DataFrame(fn_df_list) if len(fn_df_list) != 0 else empty_df

    return (
        tp_df,
        fp_df,
        tn_df,
        fn_df,
    )


def calculate_precision_recall(tp, tn, fp, fn):
    precision = (tp) / (tp + fp) if (tp + fp) != 0 else 1
    recall = (tp) / (tp + fn) if (tp + fn) != 0 else 1

    return precision, recall


# for i, user_id in enumerate(user_id_list):
for _, user_id in enumerate(sampled_users_id_list):
    print("Sampled User: {}".format(user_id))

    train_few_shots_id = train[
        (train["userId"] == user_id) & (train["rating"] >= SHOT_RATING_THRESHOLD)
    ]["movieId"].tolist()

    filtered_few_shots_df = movies_df[movies_df["movieId"].isin(train_few_shots_id)]
    watched_movie_titles = filtered_few_shots_df["title"].tolist()

    sampled_movies_df = movies_df.sample(n=SAMPLE_COUNT, random_state=42)
    sampled_movies_list = sampled_movies_df["title"].tolist()

    chatgpt_ratings = []
    exists_in_watched = []

    watched_ratings_df = ratings_df[ratings_df["userId"] == user_id]

    # print("Items: {}".format(test_movie_titles))

    for i, rows in sampled_movies_df.iterrows():
        movieId = rows["movieId"]
        title = rows["title"]

        print("Predicting rating for movie id: {} title: {}".format(movieId, title))

        prompt = RECOMMENDING_PROMPT.format(", ".join(watched_movie_titles), title)

        rating = get_rating_from_gpt(
            create_response=lambda: make_response_to_gpt(prompt)
        )

        chatgpt_ratings.append(rating)
        exists_in_watched.append(watched_ratings_df["movieId"].isin([movieId]).any())

    sampled_movies_df["predicted_ratings"] = chatgpt_ratings
    sampled_movies_df["exists_in_watched"] = exists_in_watched

    (
        tp_df_list,
        tn_df_list,
        fp_df_list,
        fn_df_list,
    ) = create_confusion_matrix(df=sampled_movies_df)

    precision, recall = calculate_precision_recall(
        tp=len(tp_df_list),
        tn=len(tn_df_list),
        fp=len(fp_df_list),
        fn=len(fn_df_list),
    )

    print("Precision: {} Recall: {}".format(precision, recall))
