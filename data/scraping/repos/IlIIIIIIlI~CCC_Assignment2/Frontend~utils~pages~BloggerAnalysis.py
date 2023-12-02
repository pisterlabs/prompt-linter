import pandas as pd
import numpy as np
import scipy.stats
from scipy.stats import norm
import altair as alt
import streamlit as st
import random
import json
from stqdm import stqdm
import openai
import requests
openai.api_key = "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"
def get_blogger_analysis(blogger, recent_posts):
    prompt = f"Blogger: {blogger}\n\n"
    prompt += f"Recent Posts: {recent_posts}\n\n"
    prompt += "Please summarize the blogger's type and the orientation of recent posts in 50 words.\n\nBlogger Analysis: "

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.5,
    )

    analysis = response.choices[0].text.strip()
    st.markdown(f"### Blogger Analysis:")
    return analysis

def Blogger_Analysis():
    # Add a checkbox to the sidebar to enable OpenAI analysis
    openai_analysis = st.sidebar.checkbox("OpenAI Analysis")

    st.write(
        """
    # Australian Blogger in Twitter v.s. Mastodon platform 📈
    This page shows the difference between the two platforms bloggers.
    """
    )
    st.markdown("---")
    # Creating a Streamlit container
    container = st.container()

    # Create two columns in the container.
    col1, col2 = container.columns(2)
    # Subtitle for Adding Columns
    col1.markdown("## 🐦 Twitter Happiness Index Australia Blogger 🇦🇺")
    col2.markdown("## 🐘 Mastodon Happiness Index Australia Blogger 🇦🇺")

    # with open('./utils/data/page2_data_T.json', encoding='utf-8') as f:
    #     twitter_data = json.load(f)
    # with open('./utils/data/page2_data_M.json', encoding='utf-8') as f:
    #     mastodon_data = json.load(f)

    # Load IP address ------------------------------------------
    with open('config.json') as f:
        localhost = json.load(f)['IP']

    # Fetch data from server
    r = requests.get(f'http://{localhost}:8000/page2data_tweet', timeout=200)
    # parse json
    twitter_data = r.json()

    # Fetch data from server
    r = requests.get(f'http://{localhost}:8000/page2data_mastodon', timeout=200)
    # parse json
    mastodon_data = r.json()
    # ----------------------------------------------------------------

    twitter_df = pd.DataFrame(twitter_data)

    # Convert columns related to numbers into integer type.
    twitter_df['retweet_count'] = twitter_df['retweet_count'].astype(int)
    twitter_df['reply_count'] = twitter_df['reply_count'].astype(int)
    twitter_df['like_count'] = twitter_df['like_count'].astype(int)
    twitter_df['quote_count'] = twitter_df['quote_count'].astype(int)

    mastodon_df = pd.DataFrame(mastodon_data)


    # Combine date and time columns into datetime
    twitter_df['datetime'] = pd.to_datetime(twitter_df['date'] + ' ' + twitter_df['time'])
    mastodon_df['datetime'] = pd.to_datetime(mastodon_df['date'] + ' ' + mastodon_df['time'])

    # Modify the sentiment column based on the emotion value
    twitter_df['sentiment'] = twitter_df['sentiment'].map({'positive': 1, 'negative': -1})
    mastodon_df['sentiment'] = mastodon_df['sentiment'].map({'positive': 1, 'negative': -1})

    twitter_groupby = twitter_df.groupby('rank')
    mastodon_groupby = mastodon_df.groupby('rank')

    # Create a slider in the sidebar to select the number of authors
    num_authors = st.sidebar.slider('Number of authors to display', min_value=1, max_value=10, value=3)
    # Add 10 st.expander in each column
    if openai_analysis:
    # Add OpenAI analysis for Twitter bloggers
        for i in stqdm(range(1, num_authors+1), st_container=st.sidebar):
            if "top" + str(i) in twitter_groupby.groups.keys():
                group = twitter_groupby.get_group('top' + str(i))
                with col1.expander(f"Top{i} 🥇Blogger", expanded=True if i <= 1 else False):
                    st.metric("Author ID", group['author_id'].values[0])
                    st.text(f"Created Time Range: {group['datetime'].min()} - {group['datetime'].max()}")

                    # metrics
                    col1_1, col1_2, col1_3 = st.columns(3)
                    with col1_1:
                        recent_sentiment_ratio = (
                                group.sort_values('datetime', ascending=False)['sentiment'].head(10) == 1).mean()
                        all_sentiment_ratio = (group['sentiment'] == 1).mean()
                        delta = (recent_sentiment_ratio - all_sentiment_ratio) / all_sentiment_ratio
                        st.metric("Positive Sentiment in recent 10 tweets", f"{recent_sentiment_ratio * 100:.2f}%",
                                  delta=f"{delta * 100:.2f}%, compared to avg. in total")
                    with col1_2:
                        st.metric("Total Like Count", group['like_count'].sum())
                    with col1_3:
                        if (group['reply_count'].sum() > group['like_count'].sum()) or (
                                group['retweet_count'].sum() > group['quote_count'].sum()):
                            st.metric("Blogger Type", "Influencer")
                        elif (group['reply_count'].sum() == group['like_count'].sum()) or (
                                group['retweet_count'].sum() == group['quote_count'].sum()):
                            st.metric("Blogger Type", "Popular")
                        else:
                            st.metric("Blogger Type", "Lovely")

                    col1_a, col1_b = st.columns(2)
                    with col1_a:
                        st.subheader("Recent 5 Positive Posts:")
                        positive_posts = group[group['sentiment'] == 1].sort_values('datetime', ascending=False)[
                            ['datetime', 'text']].head(5)
                        st.write(positive_posts.reset_index(drop=True))

                    with col1_b:
                        st.subheader("Recent 5 Negative Posts:")
                        negative_posts = group[group['sentiment'] == -1].sort_values('datetime', ascending=False)[
                            ['datetime', 'text']].head(5)
                        st.write(negative_posts.reset_index(drop=True))

                    col2_a, col2_b = st.columns(2)
                    with col2_a:
                        color = '#aaf683' if group['sentiment'].mean() > 0 else '#ff5c8a'
                        st.altair_chart(alt.Chart(group).mark_line(color=color).encode(
                            x='datetime:T',
                            y='sentiment',
                        ))

                    with col2_b:
                        group['text_len'] = group['text'].str.len()
                        text_len_df = group['text_len'].value_counts(normalize=True).reset_index().rename(columns={'index': 'text_len', 'text_len': 'proportion'})
                        text_len_df.sort_values('text_len', inplace=True)
                        st.altair_chart(alt.Chart(text_len_df).mark_line(color='#99ccff').encode(
                            x='text_len:Q',
                            y='proportion:Q',
                        ))

                    if color == '#aaf683':
                        answer = "All tweets from this blogger have a positive average sentiment."
                    else:
                        answer = "All tweets from this blogger have a negative average sentiment."
                    st.markdown(f'<span style="color: {color};">{answer}</span>', unsafe_allow_html=True)
                    recent_posts = ' '.join(group.sort_values('datetime', ascending=False)['text'].head(10).values)
                    blogger_analysis = get_blogger_analysis(group['author_id'].values[0], recent_posts)
                    st.markdown(blogger_analysis)

            if i in mastodon_groupby.groups.keys():
                group = mastodon_groupby.get_group(i)
                with col2.expander(f"Top{i} 🥇Blogger", expanded=True if i <= 1 else False):
                    st.metric("Username", group['username'].values[0])
                    st.text(f"Created Time Range: {group['datetime'].min()} - {group['datetime'].max()}")

                    # metrics
                    col2_1, col2_2, col2_3 = st.columns(3)
                    with col2_1:
                        # st.metric("Number of posts", group.shape[0])
                        recent_sentiment_ratio = (
                                group.sort_values('datetime', ascending=False)['sentiment'].head(10) == 1).mean()
                        all_sentiment_ratio = (group['sentiment'] == 1).mean()
                        # Calculate the change value
                        delta = (recent_sentiment_ratio - all_sentiment_ratio) / all_sentiment_ratio
                        st.metric("Positive Sentiment in recent 10 posts", f"{recent_sentiment_ratio * 100:.2f}%",
                                  delta=f"{delta * 100:.2f}%, compared to avg. in total")
                    with col2_2:
                        st.metric("Total Followers Count", group['followers_count'].sum())
                    with col2_3:
                        st.metric("Total Following Count", group['following_count'].sum())


                    col1_a, col1_b = st.columns(2)
                    with col1_a:
                        st.subheader("Recent 5 Positive Posts:")
                        positive_posts = group[group['sentiment'] == 1].sort_values('datetime', ascending=False)[
                            ['datetime', 'content']].head(5)
                        st.write(positive_posts.reset_index(drop=True))
                    with col1_b:
                        st.subheader("Recent 5 Negative Posts:")
                        negative_posts = group[group['sentiment'] == -1].sort_values('datetime', ascending=False)[
                            ['datetime', 'content']].head(5)
                        st.write(negative_posts.reset_index(drop=True))

                    col2_a, col2_b = st.columns(2)
                    with col2_a:
                        color = '#aaf683' if group['sentiment'].mean() > 0 else '#ff5c8a'
                        st.altair_chart(alt.Chart(group).mark_line(color=color).encode(
                            x='datetime:T',
                            y='sentiment',
                        ))
                    with col2_b:
                        group['content_len'] = group['content'].str.len()
                        content_len_df = group['content_len'].value_counts(normalize=True).reset_index().rename(columns={'index': 'content_len', 'content_len': 'proportion'})
                        content_len_df.sort_values('content_len', inplace=True)
                        st.altair_chart(alt.Chart(content_len_df).mark_line(color='#99ccff').encode(
                            x='content_len:Q',
                            y='proportion:Q',
                        ))

                    if color == '#aaf683':
                        answer = "All tweets from this blogger have a positive average sentiment."
                    else:
                        answer = "All tweets from this blogger have a negative average sentiment."
                    st.markdown(f'<span style="color: {color};">{answer}</span>', unsafe_allow_html=True)
                    recent_posts = ' '.join(group.sort_values('datetime', ascending=False)['content'].head(10).values)
                    blogger_analysis = get_blogger_analysis(group['username'].values[0], recent_posts)
                    st.markdown(blogger_analysis)
    else:
        for i in stqdm(range(1, num_authors+1), st_container=st.sidebar):
            if "top" + str(i) in twitter_groupby.groups.keys():
                group = twitter_groupby.get_group('top' + str(i))
                with col1.expander(f"Top{i} 🥇Blogger", expanded=True if i <= 1 else False):
                    st.metric("Author ID", group['author_id'].values[0])
                    st.text(f"Created Time Range: {group['datetime'].min()} - {group['datetime'].max()}")

                    # metrics
                    col1_1, col1_2, col1_3 = st.columns(3)
                    with col1_1:
                        recent_sentiment_ratio = (
                                group.sort_values('datetime', ascending=False)['sentiment'].head(10) == 1).mean()
                        all_sentiment_ratio = (group['sentiment'] == 1).mean()
                        delta = (recent_sentiment_ratio - all_sentiment_ratio) / all_sentiment_ratio
                        st.metric("Positive Sentiment in recent 10 tweets", f"{recent_sentiment_ratio * 100:.2f}%",
                                  delta=f"{delta * 100:.2f}%, compared to avg. in total")
                    with col1_2:
                        st.metric("Total Like Count", group['like_count'].sum())
                    with col1_3:
                        if (group['reply_count'].sum() > group['like_count'].sum()) or (
                                group['retweet_count'].sum() > group['quote_count'].sum()):
                            st.metric("Blogger Type", "Influencer")
                        elif (group['reply_count'].sum() == group['like_count'].sum()) or (
                                group['retweet_count'].sum() == group['quote_count'].sum()):
                            st.metric("Blogger Type", "Popular")
                        else:
                            st.metric("Blogger Type", "Lovely")

                    col1_a, col1_b = st.columns(2)
                    with col1_a:
                        st.subheader("Recent 5 Positive Posts:")
                        positive_posts = group[group['sentiment'] == 1].sort_values('datetime', ascending=False)[
                            ['datetime', 'text']].head(5)
                        st.write(positive_posts.reset_index(drop=True))

                    with col1_b:
                        st.subheader("Recent 5 Negative Posts:")
                        negative_posts = group[group['sentiment'] == -1].sort_values('datetime', ascending=False)[
                            ['datetime', 'text']].head(5)
                        st.write(negative_posts.reset_index(drop=True))

                    col2_a, col2_b = st.columns(2)
                    with col2_a:
                        color = '#aaf683' if group['sentiment'].mean() > 0 else '#ff5c8a'
                        st.altair_chart(alt.Chart(group).mark_line(color=color).encode(
                            x='datetime:T',
                            y='sentiment',
                        ))

                    with col2_b:
                        group['text_len'] = group['text'].str.len()
                        text_len_df = group['text_len'].value_counts(normalize=True).reset_index().rename(columns={'index': 'text_len', 'text_len': 'proportion'})
                        text_len_df.sort_values('text_len', inplace=True)
                        st.altair_chart(alt.Chart(text_len_df).mark_line(color='#99ccff').encode(
                            x='text_len:Q',
                            y='proportion:Q',
                        ))

                    if color == '#aaf683':
                        answer = "All tweets from this blogger have a positive average sentiment."
                    else:
                        answer = "All tweets from this blogger have a negative average sentiment."
                    st.markdown(f'<span style="color: {color};">{answer}</span>', unsafe_allow_html=True)

            if i in mastodon_groupby.groups.keys():
                group = mastodon_groupby.get_group(i)
                with col2.expander(f"Top{i} 🥇Blogger", expanded=True if i <= 1 else False):
                    st.metric("Username", group['username'].values[0])
                    st.text(f"Created Time Range: {group['datetime'].min()} - {group['datetime'].max()}")

                    # metrics
                    col2_1, col2_2, col2_3 = st.columns(3)
                    with col2_1:
                        # st.metric("Number of posts", group.shape[0])
                        recent_sentiment_ratio = (
                                group.sort_values('datetime', ascending=False)['sentiment'].head(10) == 1).mean()
                        all_sentiment_ratio = (group['sentiment'] == 1).mean()
                        # Calculate the change value
                        delta = (recent_sentiment_ratio - all_sentiment_ratio) / all_sentiment_ratio
                        st.metric("Positive Sentiment in recent 10 posts", f"{recent_sentiment_ratio * 100:.2f}%", delta=f"{delta * 100:.2f}%, compared to avg. in total")
                    with col2_2:
                        st.metric("Total Followers Count", group['followers_count'].sum())
                    with col2_3:
                        st.metric("Total Following Count", group['following_count'].sum())


                    col1_a, col1_b = st.columns(2)
                    with col1_a:
                        st.subheader("Recent 5 Positive Posts:")
                        positive_posts = group[group['sentiment'] == 1].sort_values('datetime', ascending=False)[
                            ['datetime', 'content']].head(5)
                        st.write(positive_posts.reset_index(drop=True))
                    with col1_b:
                        st.subheader("Recent 5 Negative Posts:")
                        negative_posts = group[group['sentiment'] == -1].sort_values('datetime', ascending=False)[
                            ['datetime', 'content']].head(5)
                        st.write(negative_posts.reset_index(drop=True))

                    col2_a, col2_b = st.columns(2)
                    with col2_a:
                        color = '#aaf683' if group['sentiment'].mean() > 0 else '#ff5c8a'
                        st.altair_chart(alt.Chart(group).mark_line(color=color).encode(
                            x='datetime:T',
                            y='sentiment',
                        ))
                    with col2_b:
                        group['content_len'] = group['content'].str.len()
                        content_len_df = group['content_len'].value_counts(normalize=True).reset_index().rename(columns={'index': 'content_len', 'content_len': 'proportion'})
                        content_len_df.sort_values('content_len', inplace=True)
                        st.altair_chart(alt.Chart(content_len_df).mark_line(color='#99ccff').encode(
                            x='content_len:Q',
                            y='proportion:Q',
                        ))

                    if color == '#aaf683':
                        answer = "All tweets from this blogger have a positive average sentiment."
                    else:
                        answer = "All tweets from this blogger have a negative average sentiment."
                    st.markdown(f'<span style="color: {color};">{answer}</span>', unsafe_allow_html=True)

Blogger_Analysis()
