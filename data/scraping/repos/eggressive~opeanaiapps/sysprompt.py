"""_Review classification using a system prompt_
"""
import os
import openai
import requests
import pandas as pd

openai.api_key = os.environ["OPENAI_API_KEY"]

DATA_URL = ("https://github.com/blueprints-for-text-analytics-"
            "python/blueprints-text/raw/master/data/amazon-"
            "product-reviews/reviews_5_balanced.json.gz")
LOCAL_DATA = "/home/dimitar/openai/apps/data/reviews_5_balanced.json.gz"
EXPECTED_SIZE = 16649530

if os.path.exists(LOCAL_DATA):
    local_size = os.path.getsize(LOCAL_DATA)
    if local_size == EXPECTED_SIZE:
        print(f"{LOCAL_DATA} already exists with expected size {EXPECTED_SIZE}, skipping download")
    else:
        print(f"Existing file {LOCAL_DATA} size {local_size}"
              f"does not match expected size {EXPECTED_SIZE}, re-downloading...")

        with requests.get(DATA_URL, stream=True, timeout=30) as r:
            r.raise_for_status()

            with open(LOCAL_DATA, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

else:
    print(f"{LOCAL_DATA} does not exist, downloading...")

    with requests.get(DATA_URL, stream=True, timeout=30) as r:
        r.raise_for_status()

        with open(LOCAL_DATA, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

# create data frame from the json file
df = pd.read_json(LOCAL_DATA, lines=True)
df = df.drop(columns=['reviewTime','unixReviewTime'])
df = df.rename(columns={'reviewText': 'text'})
df.sample(5, random_state=12)

# Assigning a new [1,0] target class label based on the product rating
df['sentiment'] = 'OTHER'
df.loc[df['overall'] > 3, 'sentiment'] = 'POSITIVE'
df.loc[df['overall'] < 3, 'sentiment'] = 'NEGATIVE'

# Removing unnecessary columns to keep a simple dataframe
df.drop(columns=['overall', 'reviewerID', 'summary'],
        inplace=True)
df.sample(5)

REVIEW_NUMBER = 40010
review_text = df['text'].iloc[REVIEW_NUMBER]

# Appending multiple examples of reviews in the user message,
# leaving a blank for predicted sentiment
USER_MESSAGES = ''
for review in range(13,20):
    USER_MESSAGES = USER_MESSAGES + 'Review Text: ' + df['text'].iloc[review]
    USER_MESSAGES = USER_MESSAGES + '\n' + 'Sentiment: '
    USER_MESSAGES = USER_MESSAGES + '\n'

# provide instructions to the model on what I expect it to do
SYSTEM_PROMPT = """
You will be provided with a customer review of a product on an online e-commerce website. You have to understand the context of the review
and classify the sentiment into three categories - POSITIVE or NEGATIVE or OTHER.
POSITIVE - this category indicates that the user was happy with the product and liked it
NEGATIVE - this category indicates that the user was unhappy with the product and did not like it
OTHER - please output this category if you cannot tell what the review is

Only provide the determined category and nothing else. I don't want any explanations.
"""

print ('System Prompt: ', SYSTEM_PROMPT)
print ('Review Observations:\n', USER_MESSAGES)

# make call to the OpenAI API with the prompt that we have just created
chatOutput = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_MESSAGES}
    ]
)

print ('OpenAI Response: ', chatOutput.choices[0].message.content)
