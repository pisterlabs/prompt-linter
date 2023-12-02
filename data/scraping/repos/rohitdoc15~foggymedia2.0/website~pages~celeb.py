import sys
sys.path.append('/home/rohit/news/website')
from fuzzywuzzy import fuzz
from collections import Counter

import os
import django
from django.utils import timezone
from datetime import timedelta

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'website.settings')
django.setup()

from pages.models import Video , TopPopularPersons
import openai
import time
import re

# Set your OpenAI API key
openai.api_key = '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'

# Define function to extract names
def extract_names(reply):
    names = re.findall(r'\d+\.\s+(\w+\s+\w+)', reply)
    cleaned_names = [re.sub(r'\d+\.|\.', '', name) for name in names]
    return cleaned_names

# System message to instruct the model
system_message = "name extractor which outputs person's names in format: 1. name1 2. name2 3. name3 ...."

# Initialize attempt counter and max_attempts
attempt = 1
max_attempts = 5

# Define the number of past days you want to collect data for
num_days = 2

# Prepare a list of stop words
stop_words = ["1.","Dr","Biparjoy" ,"Bengaluru", "Bengal","2.", "3.", "4.", "5.", "Singh", "Kumar", "Patel" ,"Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal", "Andaman and Nicobar Islands", "Chandigarh", "Dadra and Nagar Haveli and Daman and Diu", "The Government of NCT of Delhi", "Lakshadweep", "Puducherry", "Ladakh", "Jammu and Kashmir", "Modi" , "PM Modi" , "Cyclone Biperjoy" , "Shiv sena" , "BJP" , "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Ahmedabad", "Chennai", "Kolkata", "Surat", "Pune", "Jaipur", "Lucknow", "Kanpur", "Nagpur", "Visakhapatnam", "Indore", "Thane", "Bhopal", "Pimpri-Chinchwad", "Patna", "Vadodara", "Ghaziabad", "Ludhiana", "Coimbatore", "Agra", "Madurai", "Nashik", "Faridabad", "Meerut", "Rajkot", "Kalyan-Dombivali", "Vasai-Virar", "Varanasi", "Srinagar", "Aurangabad", "Dhanbad", "Amritsar", "Navi Mumbai", "Allahabad", "Ranchi", "Howrah", "Gwalior", "Jabalpur", "Jodhpur", "Raipur", "Kota", "Guwahati", "Chandigarh", "Thiruvananthapuram", "Solapur"]

# Iterate over each day
for day in range(num_days):
    # Calculate the 8-hour intervals for that day
    now = timezone.now() - timedelta(days=day)
    intervals = [(now - timedelta(hours=i+8), now - timedelta(hours=i)) for i in range(0, 24, 8)]

    # Open a text file to save the responses
    with open(f'extracted_names_day_{day+1}.txt', 'w') as file:
        # Iterate over the intervals
        for start, end in intervals:
            while attempt <= max_attempts:
                # Get the latest 100 videos from each channel
                channels = Video.objects.values_list('channel', flat=True).distinct()
                latest_video_titles = []
                for channel in channels:
                    latest_videos = Video.objects.filter(channel=channel, published_date__range=(start, end)).order_by('-published_date')[:5]
                    latest_video_titles.extend([video.title for video in latest_videos])

                # Concatenate all the video titles into a single string
                all_titles = ', '.join(latest_video_titles)
                print(all_titles)

                # Use OpenAI's GPT-3.5-turbo to process the text
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-16k",
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": f"I have a list of news titles : START {all_titles} END. Top 10 most talked indian person in the given news titles are:"},
                    ],
                    temperature=0.2,
                )
                
                # Extract the model's reply
                reply = completion.choices[0].message['content']

                if len(reply) <= 1000:
                    file.write(reply + "\n")
                    print(reply)
                    break
                else:
                    if attempt < max_attempts:
                        print(f"Response exceeds 160 characters. Attempt {attempt}/{max_attempts}. Waiting for 2 seconds before trying again...")
                        time.sleep(2)
                        attempt += 1
                    else:
                        print(f"Reached maximum attempts. Unable to get a response within the character limit.")
                        break

    # Now read the text file, analyze the top 5 words
    with open(f'extracted_names_day_{day+1}.txt', 'r') as file:
        data = file.read().replace("\n", " ")

    # Use regular expression to find sequences of capitalized words (treated as names)
    import re
    names = re.findall(r'(?:(?:\b[A-Z][a-z]*\b\s*)+)', data)

    # Filter names to exclude stop words
    filtered_names = [name for name in names if not any(word in name.split() for word in stop_words)]

    # Tokenize and count the words
    name_counter = Counter(filtered_names)

    # Find the 5 most common words
    top_5_names = name_counter.most_common(5)

    # Load the news titles from the day
    time_24_hours_ago = now - timedelta(hours=24)
    titles_last_24_hours = Video.objects.filter(published_date__range=(time_24_hours_ago, now)).values_list('title', flat=True)

    # Initialize a dictionary to store the name matches
    name_matches = {}

    # Iterate over the extracted top names
    for name, _ in top_5_names:
        name_matches[name] = 0

        # Iterate over the news titles
        for title in titles_last_24_hours:
            # Check if the name is present in the title
            if name.lower() in title.lower():
                name_matches[name] += 1

    # Sort the name matches in descending order
    sorted_matches = sorted(name_matches.items(), key=lambda x: x[1], reverse=True)

    print(f"Top 5 names in the news titles for day {day+1}:")
    for name, count in sorted_matches:
        print(f"{name}: {count} occurrence(s)")

     # Extract the top 3 person names and counts
    top_3_names = sorted_matches[:3]

    # Create or update the TopPopularPersons object for the date
    # Create or update the TopPopularPersons object for the date
    date_obj, created = TopPopularPersons.objects.get_or_create(date=now.date())

    # Set the person names and video counts
    top_3_names = sorted_matches[:3]
    date_obj.person1_name = top_3_names[0][0]
    date_obj.person1_video_count = top_3_names[0][1]
    date_obj.person2_name = top_3_names[1][0]
    date_obj.person2_video_count = top_3_names[1][1]
    date_obj.person3_name = top_3_names[2][0]
    date_obj.person3_video_count = top_3_names[2][1]

    # Save the object
    date_obj.save()


    print(f"Updated Top Popular Persons for day {day+1}")