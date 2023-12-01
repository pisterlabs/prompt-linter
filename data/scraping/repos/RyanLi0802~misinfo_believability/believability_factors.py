from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import pipeline
from detoxify import Detoxify
from stance_detection import stance_detection, stance_prompts
import argparse
import subprocess
import numpy as np
import pandas as pd
import openai
import scipy.stats as stats

import prompts


def get_args():
    parser = argparse.ArgumentParser(description="Believability Factors.")
    parser.add_argument("--liwc", action='store_true')
    parser.add_argument('-a', "--add_columns", nargs='+', help='add additional tweet columns')
    parser.add_argument('-r', "--rearrange", action='store_true')
    parser.set_defaults(liwc=False)
    parser.set_defaults(rearrange=False)
    return parser.parse_args()


def get_data():
    believable_by_few = pd.read_csv('believable_by_few.csv', sep=',')
    believable_by_many = pd.read_csv('believable_by_many.csv', sep=',')
    return believable_by_few, believable_by_many


def plot_distribution(few_feature, many_feature, feature_name, pval, output_dir='distributions/'):
    df = pd.DataFrame({"many": many_feature, "few": few_feature})
    ax = df.plot.hist(bins=100, alpha=0.5)
    if pval:
        plt.annotate(f"p-value = {pval}", xy=(0.5, 0.5), xycoords='axes fraction')
    plt.gca().set(title=feature_name.replace("_", " "))
    plt.savefig(output_dir + feature_name, dpi=200)
    plt.close()


def get_sentiment(df, output_path):
    if 'sentiment' in df.columns:
        return df
    print(f'running on {output_path}')
    df['sentiment'] = "";
    # print(df)
    i = 0
    for idx, row in df.iterrows():
        i += 1
        if i % 100 == 0:
            print(i)
        prompt = f'{prompts.examples}\n\nTweet:\n{row["text"]}\nSentiment:'
        try:
            res = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                max_tokens=1,
                temperature=0
            )
            df.at[idx, 'sentiment'] = res.choices[0].text.strip()
        except Exception as e:
            print(e)

    print(df[:15])

    df.to_csv(output_path, sep=',', index=False, header=True)
    print('saved to csv')
    return df


def get_personal_tone(df, output_path):
    if 'personal_tone' in df.columns:
        return df
    print(f'running on {output_path}')
    df['personal_tone'] = ''
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(idx)
        prompt = f'{prompts.personal_tone_examples}\n\nTweet:\n{row["text"]}\nIs this tweet written in a personal tone? Please look for personal appeals that include emotion, encouraging language, or colloquialisms from a community:'
        # print(prompt)
        try:
            res = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                max_tokens=1,
                temperature=0
            )
            df.at[idx, 'personal_tone'] = res.choices[0].text.strip()
        except Exception as e:
            print(row["tweetId"])
            print(e)

    df.to_csv(output_path, sep=',', index=False, header=True)
    print('saved to csv')
    print(df.head())
    return df


def get_toxicity(df):
    toxicity_model = Detoxify('unbiased')
    # toxicity_model = Detoxify('original')

    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(idx)
        results = toxicity_model.predict(row["text"])
        for label in results:
            df.at[idx, label] = results[label]
    print(df.head())
    return df


def get_topic(df, name):
    print(f'analyzing tweet topic on {name}')
    df['topic'] = ""
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(idx)
        prompt = f'{prompts.tweet_topic_examples}\n\nTweet:\n{row["text"]}\nIs the topic of this tweet politics, health, science, crime, religion, or other?'
        # print(prompt)
        try:
            res = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                max_tokens=1,
                temperature=0
            )
            df.at[idx, 'topic'] = res.choices[0].text.strip()
        except Exception as e:
            print(row["tweetId"])
            print(e)

    # df.to_csv(output_path, sep=',', index=False, header=True)
    # print('saved to csv')
    print(df.head())
    return df

def get_topic_2(df):
    df['topic2'] = ""
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(idx)
        sys_prompt = prompts.topics2_sys
        user_prompt = f'A user posted the following Tweet:\n"{row["text"]}"\n\n{prompts.topics2_user}'
        # print(user_prompt)
        # print()
        try:
            res = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0
            )
            df.at[idx, 'topic2'] = res.choices[0]['message']['content'].strip().replace('.', '')
        except Exception as e:
            print(row["tweetId"])
            print(e)
            
    # df.to_csv(output_path, sep=',', index=False, header=True)
    # print('saved to csv')
    print(df.head())
    return df

def get_narrative(df, narrative_full, narrative_abbr, narrtive_prompts):
    print(f'analyzing narrative on {narrative_full}')
    narrative_output = pd.DataFrame(columns=["text", "res", "label", "response"])
    df[narrative_abbr] = ""
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(idx)
        res, label, response = stance_detection.detect(narrative_full, row["text"], narrtive_prompts)
        # res is Yes or No token; label is 1/0, 1 for Yes and 0 for everything else
        df.at[idx, narrative_abbr] = label
        narrative_output.at[idx, "text"] = row["text"]
        narrative_output.at[idx, "res"] = res
        narrative_output.at[idx, "label"] = label
        narrative_output.at[idx, "response"] = response
    print(df.head())
    return df, narrative_output


def get_emotion(df, classifier):
    df['emotion'] = ''
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(idx)
        emotion = classifier(row["text"])[0][0]['label']
        df.at[idx, 'emotion'] = emotion
    print(df.head())
    return df


def get_emotion_scores(df):
    emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base",
                                  top_k=None)
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(idx)
        outputs = emotion_classifier(row["text"])[0]
        for output in outputs:
            df.at[idx, output['label']] = output['score']
    print(df.head())
    return df


def extend_user_data(df, tweets):
    tweets = tweets[["tweetId", "user_listed_count", "user_statuses_count", "user_verified"]]
    tweets.loc[tweets['user_verified'], 'user_verified'] = 1
    tweets.loc[tweets['user_verified'] == False, 'user_verified'] = 0
    df = df.merge(tweets, on="tweetId", how="inner")
    print(df.head())
    print(df.columns)
    return df


def add_columns(df, tweets, columns_to_add, rearrange=False):
    tweets = tweets[["tweetId"] + columns_to_add]
    # merge new columns
    df = df.merge(tweets, on="tweetId", how="inner")

    if rearrange:
        # rearrange columns so that the added columns comes first
        rest = df.columns.tolist()[2:-len(columns_to_add)]
        cols = ["tweetId", "text"] + columns_to_add + rest
        print(cols)
        df = df[cols]

    print(df.head())
    print(df.columns)
    return df


def liwc(input, name):
    print(f"running liwc-22 on {name}")

    liwcDict = "LIWC22"
    outputLocation = f"{name}_liwc.csv"
    inputFileCSV = input
    cmd_to_execute = ["LIWC-22-cli",
                      "--mode", "wc",
                      "--dictionary", liwcDict,
                      "--input", inputFileCSV,
                      "--row-id-indices", "1",
                      "--column-indices", "2",
                      "--skip-header", "yes",
                      "--output", outputLocation]
    subprocess.call(cmd_to_execute)
    print("complete!")
    print()
    print()


def process_user_verified(few, many):
    print("processing user verified...")
    few_verified = few['user_verified'].sum()
    few_unverified = len(few['user_verified']) - few_verified
    print("believable by few:")
    print("verified, unverified")
    print(few_verified, few_unverified)
    print()

    many_verified = many['user_verified'].sum()
    many_unverified = len(many['user_verified']) - many_verified
    print("believable by many:")
    print("verified, unverified")
    print(many_verified, many_unverified)

    _, pvalue = stats.fisher_exact([[many_verified, many_unverified], [few_verified, few_unverified]])
    print(f"pvalue: {pvalue}")
    print()
    print()


def process_numerical_feature(few, many, feature, output_dir='distributions/', filter=False):
    print(f"processing {feature}...")
    few_feature = few[feature]
    many_feature = many[feature]

    if filter:
        # filter out any features that doesn't meet the level of significance
        _statistic, pvalue = stats.mannwhitneyu(few_feature.to_numpy(), many_feature.to_numpy())
        if pvalue > 0.1:
            print()
            print()
            return

    few_mean = few_feature.mean()
    few_median = few_feature.median()
    few_std = few_feature.std()
    print("believable by few:")
    print("mean, median, std")
    print(few_mean, few_median, few_std)
    print()

    many_mean = many_feature.mean()
    many_median = many_feature.median()
    many_std = many_feature.std()
    print("believable by many:")
    print("mean, median, std")
    print(many_mean, many_median, many_std)

    # _statistic, pvalue = stats.ttest_ind(few_feature.to_numpy(), many_feature.to_numpy(), equal_var=False)
    # _statistic, pvalue = stats.kruskal(few_feature.to_numpy(), many_feature.to_numpy())
    _statistic, pvalue = stats.mannwhitneyu(few_feature.to_numpy(), many_feature.to_numpy())
    print(f"pvalue: {pvalue}")
    print()
    print()

    plot_distribution(few_feature, many_feature, feature, pvalue, output_dir)


def process_sentiment(few, many):
    print('processing sentiment...')
    sent_few = few['sentiment'].value_counts()
    sent_many = many['sentiment'].value_counts()
    print("believable by few:")
    print(sent_few)
    print()
    print("believable by many:")
    print(sent_many)

    sent_table = np.stack([sent_many.to_numpy(), sent_few.to_numpy()])
    _statistic, pvalue, _dof, _expected_freq = stats.chi2_contingency(sent_table)
    print(f"pvalue: {pvalue}")
    print()
    print()


def process_emotion(few, many):
    print("processing emotion...")
    emotion_few = few['emotion'].value_counts()
    emotion_many = many['emotion'].value_counts()
    print("believable by few:")
    print(emotion_few)
    print()
    print("believable by many:")
    print(emotion_many)

    for emotion in emotion_few.index:
        print(f'calculating pvalue for {emotion}')
        many_have = emotion_many[emotion]
        many_not_have = len(many['emotion']) - many_have
        few_have = emotion_few[emotion]
        few_not_have = len(few['emotion']) - few_have
        _, pvalue = stats.fisher_exact([[many_have, many_not_have], [few_have, few_not_have]])
        # print(many_have, many_not_have)
        # print(few_have, few_not_have)
        print(f"pvalue: {pvalue}")
        print()
    print()


def process_topic(few, many):
    print("processing topic...")
    topic_few = few['topic'].value_counts()
    topic_many = many['topic'].value_counts()
    print("believable by few:")
    print(topic_few)
    print()
    print("believable by many:")
    print(topic_many)

    for topic in topic_many.index:
        print(f'calculating pvalue for {topic}')
        many_have = topic_many[topic]
        many_not_have = len(many['topic']) - many_have
        few_have = topic_few.get(topic, 0)
        few_not_have = len(few['topic']) - few_have
        _, pvalue = stats.fisher_exact([[many_have, many_not_have], [few_have, few_not_have]])
        # print(many_have, many_not_have)
        # print(few_have, few_not_have)
        print(f"pvalue: {pvalue}")
        print()
    print()

def process_topic_2(few, many):
    print("processing topic 2...")
    topic_few = few['topic2'].value_counts()
    topic_many = many['topic2'].value_counts()
    print("believable by few:")
    print(topic_few)
    print()
    print("believable by many:")
    print(topic_many)
    print()


def process_narrative(narrative_abbr,  few, many):
    print("processing narrative: " + narrative_abbr)
    narra_few = few[narrative_abbr].value_counts()
    narra_many = many[narrative_abbr].value_counts()
    print("believable by few:")
    print(narra_few)
    print()
    print("believable by many:")
    print(narra_many)

    for res in narra_many.index: ## res = Yes / No
        print(f'calculating pvalue for {narrative_abbr}: {res}')
        many_have = narra_many[res]
        many_not_have = len(many[narrative_abbr]) - many_have
        few_have = narra_few.get(res, 0)
        few_not_have = len(few[narrative_abbr]) - few_have
        _, pvalue = stats.fisher_exact([[many_have, many_not_have], [few_have, few_not_have]])
        print(many_have, many_not_have)
        print(few_have, few_not_have)
        print(f"pvalue: {pvalue}")
        print()
    print()


def get_link(df):
    df['has_link'] = [1 if 'https://' in text else 0 for text in df['text']]
    # df.loc['https://' in df['text'], 'has_link'] = 1
    return df


def process_link(few, many):
    print("processing links...")
    few = get_link(few)
    many = get_link(many)
    few_link = few['has_link'].sum()
    few_no_link = len(few['has_link']) - few_link
    print("believable by few:")
    print("has link, doesn't have link")
    print(few_link, few_no_link)
    print()

    many_link = many['has_link'].sum()
    many_no_link = len(many['has_link']) - many_link
    print("believable by many:")
    print("has link, doesn't have link")
    print(many_link, many_no_link)
    _, pvalue = stats.fisher_exact([[many_link, many_no_link], [few_link, few_no_link]])
    print(f"pvalue: {pvalue}")
    print()
    print()


def process_personal_tone(few, many):
    print("processing personal tone...")
    few_personal = few['personal_tone'].value_counts()
    many_personal = many['personal_tone'].value_counts()
    print("believable by few:")
    print(few_personal)
    print()
    print("believable by many:")
    print(many_personal)

    _, pvalue = stats.fisher_exact(np.stack([many_personal.to_numpy(), few_personal.to_numpy()]))
    print(f"pvalue: {pvalue}")
    print()
    print()

def process_date(df):
    before_2022 = 0
    since_2022 = 0
    for idx, row in df.iterrows():
        year = int(row['created_at'].split()[-1])
        if year < 2022:
            before_2022 += 1
        else:
            since_2022 += 1
    print(f"before 2022: {before_2022}")
    print(f"since 2022: {since_2022}")

if __name__ == "__main__":
    openai.api_key = open('./models/.openai.key').read().replace('\n', '').replace('\r', '').strip()
    args = get_args()
    few, many = get_data()
    tweets = pd.read_csv('birdwatch-tweets-20230215.csv', low_memory=False)
    
    survey_data = pd.read_csv('survey_data.csv', sep=',')
    
    survey_data["valid"] = True
    survey_data.to_csv("survey_data.csv", sep=',', index=False, header=True)
    
    # few = add_columns(few, tweets, ['created_at'], rearrange=True)
    # many = add_columns(many, tweets, ['created_at'], rearrange=True)
    # few.to_csv("believable_by_few.csv", sep=',', index=False, header=True)
    # many.to_csv("believable_by_many.csv", sep=',', index=False, header=True)
    
    # few = get_sentiment(few, 'believable_by_few.csv')
    # many = get_sentiment(many, 'believable_by_many.csv')
    # few = get_personal_tone(few, 'believable_by_few.csv')
    # many = get_personal_tone(many, 'believable_by_many.csv')   
    
    if 'user_verified' not in few.columns:
        # tweets = tweets.rename(columns = {'id': 'tweetId'})
        few = extend_user_data(few, tweets)
        many = extend_user_data(many, tweets)
        few.to_csv("believable_by_few.csv", sep=',', index=False, header=True)
        many.to_csv("believable_by_many.csv", sep=',', index=False, header=True)

    if 'emotion' not in few.columns:
        emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base",
                                      top_k=1)
        few = get_emotion(few, emotion_classifier)
        many = get_emotion(many, emotion_classifier)
        few.to_csv("believable_by_few.csv", sep=',', index=False, header=True)
        many.to_csv("believable_by_many.csv", sep=',', index=False, header=True)

    if 'joy' not in few.columns:
        few = get_emotion_scores(few)
        many = get_emotion_scores(many)
        few.to_csv("believable_by_few.csv", sep=',', index=False, header=True)
        many.to_csv("believable_by_many.csv", sep=',', index=False, header=True)

    if args.liwc:
        liwc("believable_by_few.csv", "believable_by_few")
        liwc("believable_by_many.csv", "believable_by_many")

    if args.add_columns:
        # few = add_columns(few, tweets, args.add_columns, rearrange=args.rearrange)
        # many = add_columns(many, tweets, args.add_columns, rearrange=args.rearrange)
        # few.to_csv("believable_by_few.csv", sep=',', index=False, header=True)
        # many.to_csv("believable_by_many.csv", sep=',', index=False, header=True)
        survey_data = add_columns(survey_data, tweets, args.add_columns, rearrange=args.rearrange)
        survey_data.to_csv("survey_data.csv", sep=',', index=False, header=True)
        
    # if 'topic' not in few.columns:
    #     few = get_topic(few, "believable by few")
    #     many = get_topic(many, "believable by many")
    #     few.to_csv("believable_by_few.csv", sep=',', index=False, header=True)
    #     many.to_csv("believable_by_many.csv", sep=',', index=False, header=True)
    #     print('saved to csv')
    
    # if 'toxicity' not in few.columns:
    #     few = get_toxicity(few)
    #     many = get_toxicity(many)
    #     few.to_csv("believable_by_few.csv", sep=',', index=False, header=True)
    #     many.to_csv("believable_by_many.csv", sep=',', index=False, header=True)
    #     print('saved to csv')
    
    if 'topic2' not in few.columns:
        few = get_topic_2(few)
        many = get_topic_2(many)
        few.to_csv("believable_by_few.csv", sep=',', index=False, header=True)
        many.to_csv("believable_by_many.csv", sep=',', index=False, header=True)
        print('saved to csv')

    # narrative detection
    for narra_abbr, narra_value in stance_prompts.narratives.items():
        if narra_abbr not in few.columns:
            narra_full, narra_prompts = narra_value
            few, few_response = get_narrative(few, narra_full, narra_abbr, narra_prompts)
            many, many_response = get_narrative(many, narra_full, narra_abbr, narra_prompts)
            few.to_csv("believable_by_few.csv", sep=',', index=False, header=True)
            many.to_csv("believable_by_many.csv", sep=',', index=False, header=True)
            few_response.to_csv("stance_detection/" + narra_abbr + "_few_response.csv", sep=',', index=False, header=True)
            many_response.to_csv("stance_detection/" + narra_abbr + "_many_response.csv", sep=',', index=False, header=True)
            print('saved to csv')



    # # get the 80% threshold of user followers and retweets
    # for feature in ["user_followers_count", "retweet_count"]:
    #     few_feature = few[feature]
    #     many_feature = many[feature]
    #     combined_feature = pd.concat([few_feature, many_feature], ignore_index=True)

    #     # getting the 80th percentile
    #     threshold = combined_feature.quantile(0.8)
    #     print(f"80%% threshold of {feature}: {threshold}")
    
    # # user features
    # print("processing user features...")
    # process_user_verified(few, many)
    # user_features = ['user_followers_count','user_friends_count','user_favourites_count','user_listed_count','user_statuses_count']
    # for feature in user_features:
    #     process_numerical_feature(few, many, feature)
        
    # # tweet features
    # print("processing tweet features")
    # tweet_features = ['favorite_count', 'retweet_count']
    # for feature in tweet_features:
    #     process_numerical_feature(few, many, feature)
    
    # # content data
    # print("processing content features...")
    # process_sentiment(few, many)
    # process_topic(few, many)
    # # process_emotion(few, many)
    # process_link(few, many)
    # process_personal_tone(few, many)
    # process_topic_2(few, many)
    
    process_date(survey_data)
    
    # print("processing user features...")
    # user_features = ['user_followers_count','user_friends_count','user_favourites_count','user_listed_count','user_statuses_count']
    # for feature in user_features:
    #     process_numerical_feature(survey_data, survey_data, feature)
        
    # print("processing tweet features")
    # tweet_features = ['favorite_count', 'retweet_count']
    # for feature in tweet_features:
    #     process_numerical_feature(survey_data, survey_data, feature)
    
    # # processing each emotion as numerical data
    # emotions = ["neutral", "fear", "anger", "surprise", "sadness", "joy", "disgust"]
    # for emotion in emotions:
    #     process_numerical_feature(few, many, emotion, "distributions/emotions/")
        
    # toxicity_features = ["toxicity", "severe_toxicity", "obscene", "identity_attack", "insult", "threat", "sexual_explicit"]
    # for toxicity_feature in toxicity_features:
    #     process_numerical_feature(few, many, toxicity_feature, "distributions/toxicity/")
    
    # # liwc
    # print("processing liwc features")
    # liwc_few = pd.read_csv('believable_by_few_liwc.csv', sep=',')
    # liwc_many = pd.read_csv('believable_by_many_liwc.csv', sep=',')
    # # liwc_features = ["Analytic", "Clout", "Authentic", "affiliation", "achieve", "power", "moral", 
    # #                  "conflict", "ppron", "allnone", "swear", "allure", "curiosity", "insight", "cause", "discrep", "tentat", "certitude", "differ"]
    # liwc_features = liwc_many.columns.tolist()[2:]
    # for feature in liwc_features:
    #     process_numerical_feature(liwc_few, liwc_many, feature, "distributions/LIWC/", True)
    
    # narrative detection
    for narra_abbr, _ in stance_prompts.narratives.items():
        process_narrative(narra_abbr, few, many)
        narra_full, narra_prompts = narra_value
        print('saved to csv')
    
    print("all done!")
