import sys
import time
import os
import django
from django.utils import timezone
from datetime import timedelta
from fuzzywuzzy import fuzz
import openai
from googletrans import Translator

sys.path.append('/home/rohit/news/website')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'website.settings')
django.setup()
stopwords = {'hindi news', 'world news', 'r bharat' , 'Gravitas', 'Top Headlines',  'ABP News' , 'Political controversies' , 'India News', 'Breaking News'}
from pages.models import NewsChannel, Video, TrendingTopic


def filter_quotes(topics):
    return [topic.replace("'", "") for topic in topics]


def translate_to_english(text):
    translator = Translator()
    result = translator.translate(text, dest='en')
    return result.text


def read_topics_from_file(filename):
    with open(filename, 'r') as file:
        topics = file.read().splitlines()
    return topics


def update_topics_file(filename, topics):
    with open(filename, 'w') as file:
        for topic in topics:
            file.write(f"{topic}\n")


def match_with_trending_topics(videos, topics, stopwords):
    updated_videos_count = 0
    for video in videos:
        title = video.title
        if any(chr.isalpha() for chr in title if ord(chr) > 128):
            title = translate_to_english(title)

        best_score = 0
        best_topic = None

        for topic in topics:
            if topic.lower() in stopwords:
                continue

            similarity = fuzz.token_set_ratio(title, topic)

            if similarity > best_score and similarity > 50:
                best_score = similarity
                best_topic = topic

        if best_topic:
            video.topic = best_topic
            video.save()
            updated_videos_count += 1

    return updated_videos_count


def get_popular_topics(channel, days=3):
    time_days_ago = timezone.now() - timedelta(days=days)
    videos = Video.objects.filter(channel=channel, published_date__gte=time_days_ago)
    topics = {video.topic for video in videos}
    return trim_topics(topics)


def trim_topics(topics):
    trimmed_topics = []
    for topic in topics:
        words = topic.split()
        if len(words) > 4:
            trimmed_topic = " ".join(words[:4])
            trimmed_topics.append(trimmed_topic)
        else:
            trimmed_topics.append(topic)
    return trimmed_topics


def remove_similar_topics(topics):
    distinct_topics = set(topics.copy())  # Convert topics to a set to ensure distinct values
    for topic1 in topics:
        for topic2 in distinct_topics.copy():  # Use a copy of distinct_topics to iterate and modify it
            if topic1 != topic2:
                similarity = fuzz.token_set_ratio(topic1, topic2)
                if similarity > 80:
                    if len(topic1) > len(topic2):
                        distinct_topics.discard(topic1)
                    else:
                        distinct_topics.discard(topic2)
                    break
    return list(distinct_topics)  # Convert distinct_topics back to a list before returning



def remove_stopwords(topics, stopwords):
    return [topic for topic in topics if topic.lower() not in stopwords]


time_12_hours_ago = timezone.now() - timedelta(hours=12)
channels = NewsChannel.objects.all()

for channel in channels:
    channel_name = channel.name
    old_topics = read_topics_from_file(f'{channel_name}.txt')
    old_topics = trim_topics(old_topics)  # Trim the topics as soon as they are read from the file
    update_topics_file(f'{channel_name}.txt', old_topics)  # Update the file with trimmed topics
    old_topics = filter_quotes(old_topics)  # Add this line to filter out single quotes
    old_topics = remove_similar_topics(old_topics)
    old_topics = remove_stopwords(old_topics, stopwords)  # Remove stopwords from existing topics

    popular_topics = get_popular_topics(channel, days=3)

    common_topics = set(old_topics).intersection(popular_topics)
    old_topics = [topic for topic in old_topics if topic not in common_topics]
    old_topics.extend(common_topics)

    update_topics_file(f'{channel_name}.txt', old_topics)

    recent_videos = Video.objects.filter(channel=channel, published_date__gte=time_12_hours_ago)
    updated_videos_count = match_with_trending_topics(recent_videos, old_topics, stopwords)

    print(f"\nSummary for channel '{channel_name}':")
    print(f"Number of videos processed: {recent_videos.count()}")
    print(f"Number of videos updated with new topics: {updated_videos_count}\n")
