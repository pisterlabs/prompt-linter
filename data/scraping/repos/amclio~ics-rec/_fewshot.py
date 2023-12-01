from data import train, test, user_id_list, ratings_df, movies_df
from random import sample
import os
import openai
import re
from retry import retry

openai.api_key = os.getenv("OPENAI_API_KEY")

SHOT_RATING_THRESHOLD = 5
ACUAL_RATING_THRESHOLD = 3
PREDICTED_RATING_THRESHOLD = 5

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


def create_confusion_matrix(actual_ratings, predicted_ratings):
    tp_indices_list = []
    tn_indices_list = []
    fp_indices_list = []
    fn_indices_list = []

    for i, actual_rating in enumerate(actual_ratings):
        if actual_rating >= ACUAL_RATING_THRESHOLD:
            if predicted_ratings[i] >= PREDICTED_RATING_THRESHOLD:
                tp_indices_list.append(i)
            else:
                fn_indices_list.append(i)
        elif actual_rating < ACUAL_RATING_THRESHOLD:
            if predicted_ratings[i] < PREDICTED_RATING_THRESHOLD:
                tn_indices_list.append(i)
            else:
                fp_indices_list.append(i)

    return tp_indices_list, tn_indices_list, fp_indices_list, fn_indices_list


def calculate_precision_recall(tp, tn, fp, fn):
    precision = (tp) / (tp + fp)
    recall = (tp) / (tp + fn)

    return precision, recall


# for i, user_id in enumerate(user_id_list):
for _, user_id in enumerate(sampled_users_id_list):
    print("Sampled User: {}".format(user_id))

    train_few_shots_id = train[
        (train["userId"] == user_id) & (train["rating"] >= SHOT_RATING_THRESHOLD)
    ]["movieId"].tolist()

    filtered_few_shots_df = movies_df[movies_df["movieId"].isin(train_few_shots_id)]
    watched_movie_titles = filtered_few_shots_df["title"].tolist()

    test_user_ratings_df = test[test["userId"] == user_id]
    test_user_movie_indices = test_user_ratings_df["movieId"].tolist()
    filtered_test_df = movies_df[movies_df["movieId"].isin(test_user_movie_indices)]
    test_movie_titles = filtered_test_df["title"].tolist()

    chatgpt_ratings = []

    print("Items: {}".format(test_movie_titles))

    for i, title in enumerate(test_movie_titles):
        prompt = RECOMMENDING_PROMPT.format(
            ", ".join(watched_movie_titles), test_movie_titles[i]
        )

        response = make_response_to_gpt(prompt)

        message = response.choices[0].message.content  # type: ignore
        print(message)

        match = re.search(r"`(\d+)`", message)
        rating = None

        if match:
            rating_string = match.group(1)
            rating = int(rating_string)

            chatgpt_ratings.append(rating)
        else:
            print("No match found for the item: {}".format(title))
            chatgpt_ratings.append(None)

    test_ratings = test_user_ratings_df["rating"].tolist()

    print("Test Ratings: {}".format(test_ratings))
    print("ChatGPT Ratings: {}".format(chatgpt_ratings))

    (
        tp_indices_list,
        tn_indices_list,
        fp_indices_list,
        fn_indices_list,
    ) = create_confusion_matrix(
        actual_ratings=test_ratings, predicted_ratings=chatgpt_ratings
    )

    precision, recall = calculate_precision_recall(
        tp=len(tp_indices_list),
        tn=len(tn_indices_list),
        fp=len(fp_indices_list),
        fn=len(fn_indices_list),
    )

    print("Precision: {} Recall: {}".format(precision, recall))
