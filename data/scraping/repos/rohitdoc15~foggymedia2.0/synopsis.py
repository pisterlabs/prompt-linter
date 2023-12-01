import sys
sys.path.append('/home/rohit/news/website')
from googletrans import Translator
import os
import django
from django.utils import timezone
from datetime import timedelta

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'website.settings')
django.setup()
from collections import Counter as CollectionsCounter
from pages.models import NewsChannel, Video, TrendingTopic
import openai
import time

# Configure OpenAI API credentials
openai.api_key = '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'

# Get the current time
now = timezone.now()

# Get the time 4 hours ago
time_4_hours_ago = now - timedelta(hours=4)

# Fetch all distinct topics from the database
topics = TrendingTopic.objects.values_list('topic', flat=True).distinct()

# Translate function using translate package
def translate_text(text):
    translator = Translator()
    result = translator.translate(text, dest='en')
    return result.text

# Iterate over each topic
for topic in topics:
    # Fetch the last 15 video titles of the given topic
    videos = Video.objects.filter(topic=topic).order_by('-published_date')[:15]
    titles = [video.title for video in videos]

    # Translate the titles using the translate package
    translated_titles = [translate_text(title) for title in titles]

    # Concatenate the translated titles into a single string
    titles_text = '\n'.join(translated_titles)
    completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                        {"role": "system", "content": "sarcastic news summeriser"},
                        {"role": "user", "content": f"Generate a sarcastic summary for the topic: {topic}\nTitles:\n{titles_text}. summary should be in 2-3 line"},
                ],
                
            )
    # Extract the model's reply
    synopsis = completion.choices[0].message['content']




    # Update the synopsis in the database
    trending_topic = TrendingTopic.objects.get(topic=topic)
    trending_topic.synopsis = synopsis
    trending_topic.save()

    print(f"Generated synopsis for topic '{topic}': {synopsis}")
