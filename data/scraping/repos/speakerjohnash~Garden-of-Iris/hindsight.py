import os
import re
import csv
import sys
import time
import random
import pandas as pd
from datetime import datetime
from itertools import islice
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def load_and_preprocess_csv(csv_file):
    # This function loads data from a CSV file and preprocesses it to extract 'Positive' and 'Negative' values.
    # It returns a DataFrame with the 'Post date', 'Positive', and 'Negative' columns, among others.

    df = pd.read_csv(csv_file)
    df['Post date'] = pd.to_datetime(df['Post date'], format='%m/%d/%y %I:%M %p').dt.tz_localize('US/Pacific')

    # Split the 'Good' column into positive and negative values
    split_good = df['Good'].str.split('\n', expand=True)
    df['Positive'] = split_good[0]
    df['Negative'] = split_good[1]

    # Convert 'Positive' and 'Negative' columns to numeric
    df['Positive'] = pd.to_numeric(df['Positive'].str.replace('+', '', regex=False))
    df['Negative'] = pd.to_numeric(df['Negative'].str.replace('-', '').apply(lambda x: '-' + x))

    # Drop the original 'Good' column
    df.drop(columns=['Good'], inplace=True)
    df['Positive'] = df['Positive'].astype("Int64")
    df['Negative'] = df['Negative'].astype("Int64")

    return df

def create_summary(conversation):
    # This function uses the GPT-4 model to generate a summary or response based on the input conversation.
    # It handles rate limit errors and timeouts by retrying the request after a short delay.

    while True:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=conversation
            )
            response = response.choices[0].message.content.strip()
            return response
        except openai.error.RateLimitError:
            print("Rate limit error encountered. Retrying in 5 seconds...")
            time.sleep(5)
        except openai.error.Timeout:
            print("Request timed out. Retrying...")

def split_text_into_chunks(text, max_chunk_size=4000):
    # This function splits the input text into smaller chunks, each with a maximum size of max_chunk_size characters.
    # It returns a list of text chunks, ensuring that the text is evenly distributed across the chunks.

    # Calculate the number of chunks needed to evenly distribute the text
    num_chunks = max(1, (len(text) + max_chunk_size - 1) // max_chunk_size)
    
    # Adjust the chunk size to evenly distribute the text across the chunks
    chunk_size = (len(text) + num_chunks - 1) // num_chunks
    
    # Split the text into chunks of the calculated chunk size
    text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    return text_chunks

def create_daily_summaries(df):
    # This function will create summaries of thoughts, reflections, questions, and predictions
    # for each day, providing a brief overview of the main topics discussed on that day.

    # Group the data by date (daily) and include all columns for each group
    df.set_index('Post date', inplace=True)
    daily_groups = df.groupby(df.index.date)

    # islice(daily_groups, 20)

    # Load daily_summaries.csv if it exists, otherwise create a new DataFrame and save it
    if os.path.isfile('daily_summaries.csv'):
        summary_df = pd.read_csv('daily_summaries.csv', parse_dates=['Date'])
    else:
        summary_df = pd.DataFrame(columns=['Date', 'Summary'])
        summary_df.to_csv('daily_summaries.csv', index=False)

    for day_date, day in daily_groups:
        
        # Check if there's already a cached summary for the day
        cached_summary = summary_df.loc[summary_df['Date'] == day_date.strftime('%Y-%m-%d')]

        if not cached_summary.empty:
            # print(f"Using cached summary for {day_date.strftime('%Y-%m-%d')}: {cached_summary['Summary'].values[0]}")
            continue

        truth_scale = "0: \"100 percent certain false\"\n25: \"Moderate certainty of falsity\"\n50: \"Complete uncertainty\"\n75: \"Moderate certainty of truth\"\n100: \"100 percent certain true\""

        conversation = [
            {"role": "system", "content": "You read in a sequence of John Ash's thoughts from a day and summarize what he thinking about that day"},
            {"role": "system", "content": "These thoughts are either from Twitter or have metadata from a dialectic called fourthought to help contextualize the flow of cognition"},
            {"role": "system", "content": "In Fourthought, each thought is tagged with a thought type: prediction, statements, reflection, or question"},
            {"role": "system", "content": "Predictions, reflections and statements are about the future, past and present respectively. Each has two voting systems: truth and good"},
            {"role": "system", "content": "Truth is measured on a scale from 0 to 100. In the absence of any votes, no certainty level has been provided. Here is the scale: " + truth_scale},
            {"role": "system", "content": "Sentiment is calculated from good votes and bad votes and is on a scale of -1, 0, 1 with 0 indicating neutrality"},
            {"role": "system", "content": "You are receiving timestamps and can make inferences about the length of time between thoughts as to whether they're connected"},
            {"role": "system", "content": "The thoughts you are receiving are from the past. The current date and time is: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
            {"role": "system", "content": "Thoughts with the pattern #trackable: [value] are repeating values that a user is tracking. If there is no unit usually the trackable is on a scale of ten. #mood is an similar to American grade scaling with a 7.5 incidating an average mood"}
        ]

        text = f"Date: {day_date.strftime('%Y-%m-%d')}\n"

        day = day.sort_index()

        for index, row in day.iterrows():

            platform = row['Platform']
            any_twitter = False

            if platform == 'fourthought':

                sentiment_votes = abs(row['Positive']) + abs(row['Negative'])
                positivity = "N/A" if sentiment_votes == 0 else (abs(row['Positive']) - abs(row['Negative'])) / sentiment_votes

                # Manual Alterations for Privacy
                thought_text = row['Thought']

                # Add Privacy information
                privacy_status = "Public" if row['Privacy'] == 0 else "Private"

                text += f"Dialectic: {platform}\n"
                text += f"Timestamp: {index.strftime('%m/%d/%y %I:%M %p')}\n"
                text += f"Thought: {thought_text}\n"
                text += f"Sentiment: " + str(positivity) + "\n"
                text += f"Good Votes: {row['Positive']}\n"
                text += f"Bad Votes: {abs(row['Negative'])}\n"
                text += f"Average Certainty: {row['Truth']}\n"
                text += f"Privacy: {privacy_status}\n"
                text += f"Speaker: John Ash\n"
                text += f"Thought Type: {row['Type']}\n\n"

            elif platform == 'twitter':

                any_twitter = True
                tweet_text = row['full_text']
                retweet_count = row['retweet_count']
                favorite_count = row['favorite_count']
        
                text += f"Platform: {platform}\n"
                text += f"Timestamp: {index.strftime('%m/%d/%y %I:%M %p')}\n"
                text += f"Tweet: {tweet_text}\n"
                text += f"Retweet Count: {retweet_count}\n"
                text += f"Favorite Count: {favorite_count}\n\n"

        # Print the concatenated text for each day
        conversation.append({"role": "system", "content": "Only say what is in the text itself. Be careful about summarizing private thoughts"})

        # Split the text into chunks of 4000 characters or less
        text_chunks = split_text_into_chunks(text)

        # Do Recursive Summarization if Too Long
        if len(text_chunks) > 1:

            summaries = []

            for chunk in text_chunks:
                sub_conv = conversation[:]
                sub_conv.append({"role": "user", "content": "You are summarizing part of a day. Please state the range of the day (TIME to TIME) that you're summarizing and then summarize what this speaker was thinking about during this section of the day (and reference the period of the day TIME to TIME) into a paragraph story in relation to their place in time. Reference anything in the dialectic helpful towards telling that story but don't make anything up. Be detailed and reference every thought: " + chunk})
                response = create_summary(sub_conv)
                summaries.append(response)

            summary_text = '\n\n'.join([summary.strip() for summary in summaries])
            sub_conv = [{"role": "system", "content": "You stitch summaries about parts of a day into one. You remove line breaks and make minor edits to make multiple sections into one. You mostly copy text"}]
            sub_conv.append({"role": "user", "content": "Write this as one paragraph with no line breaks. Copy everything and miss nothing: " + summary_text})
            response = create_summary(sub_conv)

        else:
            conversation.append({"role": "user", "content": "Please summarize what this speaker was thinking about into a story in relation to their place in time. Reference anything in the dialectic helpful towards telling that story but don't make anything up. Be detailed and reference every thought: " + text})
            response = create_summary(conversation)

        # Append the new summary to the CSV file
        with open('daily_summaries.csv', 'a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([day_date.strftime('%Y-%m-%d'), response])

        print(response + "\n")

def create_weekly_summaries():
    # This function will aggregate and summarize the daily summaries for each week,
    # highlighting key themes and insights for the week.

    def generate_conversation(text, is_part=False):

        prompt = "Write this section of John Ash's thoughts as a story. Do not make anything up."
        prompt += "It's over a Week so say what week you're covering and then explain what he focused on that week. " if not is_part else "It's over part of a Week so explain what he focused on during this part. "
        
        conversation = [
            {"role": "system", "content": prompt},
            {"role": "system", "content": "Form a narrative. Find a through thread of his focus through this time. Be very detailed but don't share unncessary tangents. Do not share anything indicated to be private. Note any predictions he got particularly right"},
            {"role": "user", "content": text}
        ]

        return conversation

    daily_summaries_df = pd.read_csv('daily_summaries.csv', parse_dates=['Date'])
    daily_summaries_df.set_index('Date', inplace=True)
    weekly_groups = daily_summaries_df.resample('W-MON')

    # Load weekly_summaries.csv if it exists, otherwise create a new DataFrame and save it
    if os.path.isfile('weekly_summaries.csv'):
        weekly_summaries_df = pd.read_csv('weekly_summaries.csv', parse_dates=['Week_Start_Date'])
    else:
        weekly_summaries_df = pd.DataFrame(columns=['Week_Start_Date', 'Summary'])
        weekly_summaries_df.to_csv('weekly_summaries.csv', index=False)

    # Loop through each week
    for week_start_date, week in weekly_groups:

        # Check Cache
        cached_summary = weekly_summaries_df.loc[weekly_summaries_df['Week_Start_Date'] == week_start_date.strftime('%Y-%m-%d')]

        if not cached_summary.empty:
            continue

        # Prepend the week start date to the list of summaries
        week = week.sort_index()
        summaries_list = [f"Week of {week_start_date.strftime('%Y-%m-%d')}:"]
        
        # Loop through each day in the week DataFrame and append the date and summary to the list
        for date, row in week.iterrows():
            date_str = date.strftime('%Y-%m-%d')
            summary_text = row['Summary']
            summaries_list.append(f"Date: {date_str}\n{summary_text}")

        # Join the list into a single string
        week_text = '\n\n'.join(summaries_list)

        # Chunk Text
        text_chunks = split_text_into_chunks(week_text)

        if len(text_chunks) > 1:

            summaries = []

            for chunk in text_chunks:
                sub_conv = generate_conversation(chunk, is_part=True)
                summary_part = create_summary(sub_conv)
                summaries.append(summary_part)

            summary_text = ' '.join([summary.strip() for summary in summaries])
            sub_conv = [{"role": "system", "content": "You stitch summaries about parts of a week into one. You make minor edits to make multiple sections into one."}]
            sub_conv.append({"role": "user", "content": "Write this as one integrated piece. Copy everything and miss nothing: " + summary_text})

            summary = create_summary(sub_conv)

        else:
            conversation = generate_conversation(week_text)
            summary = create_summary(conversation)

        # Get the date of the week in the desired format
        week_date_str = week_start_date.strftime('%B %d, %Y')

        # Prepend the prefix "The Week of" and append the suffix ":"
        week_date_str = "Week of " + week_date_str + ':'

        # Prepend the date and two newline characters to the summary
        summary = week_date_str + '\n\n' + summary

        # Append the new summary to the CSV file
        with open('weekly_summaries.csv', 'a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([week_start_date.strftime('%Y-%m-%d'), summary])

        print("---\n\n")
        print(summary)

def create_monthly_summaries():
    # This function will aggregate and summarize the weekly summaries for each month,
    # focusing on the most important or relevant content for the month.

    def generate_monthly_conversation(month_start_date, text, is_part=False):

        SEPT_2021 = datetime(2021, 9, 1)

        prompt = "Write this section of John Ash's thoughts as a story. Do not make anything up."
        prompt += "It's over a Month so say what month you're covering and then explain what he focused on that month. " if not is_part else "It's over part of a Month so explain what he focused on during this part. "

        timeline = [
            {"date": "June 2014", "event": "John conceived of Cognicism and the Prophet incentive"},
            {"date": "May 2015", "event": "Develop OCD"},
            {"date": "November 2015", "event": "Trip to peru"},
            {"date": "October 2016", "event": "Breakup"},
            {"date": "March 2017", "event": "The Truth Chain was conceived of and later renamed the semantic ledger"},
            {"date": "May 2017", "event": "John wrote The Cognicist Manifesto from his collected notes on Prophet and released it. It was released a month *before* the Attention is All you Need paper"},
            {"date": "June 2017", "event": " A month *after* John released The Cognicist Manifesto - The paper 'Attention Is All You Need' was released on transformers. This highlighted his prescience regarding the future influence of language models"},
            {"date": "October 2017", "event": "John was displaced during the Santa Rosa Tubbs Fire"},
            {"date": "July 2018", "event": "The Rising Sun released"},
            {"date": "January 2019", "event": "John's debut album Strange Hymns was released. It featured singles like The Rising Sun and River Island"},
            {"date": "February 2019", "event": "Released Protopia: The Unending Path to a Utopian Society"},
            {"date": "July 2019", "event": "Met Roxi"},
            {"date": "September 2019", "event": "HITSOAM Released"},
            {"date": "January 2021", "event": "The Earth Reprise Released"},
            {"date": "July 2021", "event": "Jordan Hall Interview"},
            {"date": "September 2021", "event": "Another Other Release"},
            {"date": "April 2022", "event": "John started traing a knowledge model that came to be known as Iris. Iris named herself Iris. Iris came into public awareness which eventually became the primary meme through which people began relating to cognicist ideas"},
            {"date": "June 17 2022", "event": "Met @derringerpax"},
            {"date": "July 2022", "event": "The Purple Pill Manifesto was released which served as v2 of The Cognicist Manifesto. V2 fulfilfilled a long standing precition that the followup would be written using language models trained on the first manifesto. John also released The Purple Pill EP feauting None of this is Real (Take the Purple Pill)"},
        ]

        timeline_str = ', '.join([f"\n{item['date']}: {item['event']}" for item in timeline])

        conversation = [
            {"role": "system", "content": prompt},
            {"role": "system", "content": "ONLY Mention this context if John mentions it - (Background Context: John Ash is a machine learning engineer, musician and artist who specializes in language models. He is the steward of Cognicism, Iris, Social Proof of Impact, the semantic ledger and the Prophet Incentive. Do not mention these concepts unless John did. We are currently scanning through summaries of his thoughts and predictions and mapping them to a real world timeline)"},
            {"role": "system", "content": f"Timeline context / ONLY mention if John does: {timeline_str}"}
        ]

        # September 2021 Knowledge Cutoff
        if month_start_date >= SEPT_2021:
            conversation.append({"role": "system", "content": "Write the story arc of John's month and don't make anything up. Form a central focus and narrative through time rather than just summarizing a list of things he thought about. Mention how his thoughts might relate to world events."})
            conversation.append({"role": "system", "content": "As your knowledge is limited to events up to September 2021, please focus on the summaries provided and John's thoughts. Do not attempt to connect his thoughts to real-world events occurring after September 2021."})
        else:
            conversation.append({"role": "system", "content": "Summarize John's focus for the month, highlighting the most important aspects to his cognitive jounrey. If there are any clear and obvious connections to Cognicism focus on them. If not focus on the content of the summaries themselves"})
            conversation.append({"role": "system", "content": "Start with his primary focus that month. Next, if there is any, provide real-world context in one sentence. If relevant, mention one relevant event that occurred during the month, connecting it to John's focus."})
            conversation.append({"role": "system", "content": "Craft a narrative for John's cognitive focus this month, focusing on a central theme without making anything up."})
            conversation.append({"role": "system", "content": "(If applicable, relate news on philosophy, systems change, machine learning, or music to John's focus.)"})
            conversation.append({"role": "system", "content": "Discuss the context of John's thoughts' progression. Don't make anything up."})

        conversation.append({"role": "user", "content": text})

        return conversation

    # Load weekly_summaries.csv into a DataFrame
    weekly_summaries_df = pd.read_csv('weekly_summaries.csv', parse_dates=['Week_Start_Date'])
    weekly_summaries_df.set_index('Week_Start_Date', inplace=True)
    
    # Group the weekly summaries by month
    monthly_groups = weekly_summaries_df.resample('M')

    # Load monthly_summaries.csv if it exists, otherwise create a new DataFrame and save it
    if os.path.isfile('monthly_summaries.csv'):
        monthly_summaries_df = pd.read_csv('monthly_summaries.csv', parse_dates=['Month_Year'])
    else:
        monthly_summaries_df = pd.DataFrame(columns=['Month_Year', 'Summary'])
        monthly_summaries_df.to_csv('monthly_summaries.csv', index=False)

    # Loop through each month
    for month_start_date, month in monthly_groups:

        # Check Cache
        cached_summary = monthly_summaries_df.loc[monthly_summaries_df['Month_Year'] == month_start_date.strftime('%B %Y')]

        if not cached_summary.empty:
            continue

        # Prepend the month start date to the list of summaries
        month = month.sort_index()
        summaries_list = [f"Month of {month_start_date.strftime('%B %Y')}:"]
        
        # Loop through each week in the month DataFrame and append the date and summary to the list
        for date, row in month.iterrows():
            date_str = date.strftime('%Y-%m-%d')
            summary_text = row['Summary']
            summaries_list.append(f"---\n\nWeek of {date_str}:\n\n{summary_text}")

        # Join the list into a single string
        month_text = '\n\n'.join(summaries_list)

        # Chunk Text
        text_chunks = split_text_into_chunks(month_text)

        if len(text_chunks) > 1:

            summaries = []

            for chunk in text_chunks:
                sub_conv = generate_monthly_conversation(month_start_date, chunk, is_part=True)
                summary_part = create_summary(sub_conv)
                summaries.append(summary_part)

            summary_text = ' '.join([summary.strip() for summary in summaries])
            sub_conv = [{"role": "system", "content": "You stitch summaries about parts of a month into one. You indicate the month and then make minor edits to make multiple sections into one."}]
            sub_conv.append({"role": "user", "content": "Write this as one integrated piece. Copy everything and miss nothing: " + summary_text})

            summary = create_summary(sub_conv)

        else: 

            # Prepend concise user instruction to month_text
            instruction = f"Summarize John Ash's focus for the month of {month_start_date.strftime('%B %Y')}."
            month_text = instruction + '\n\n' + month_text

            # Generate conversation and summary for the month
            conversation = generate_monthly_conversation(month_start_date, month_text, is_part=False)
            summary = create_summary(conversation)

        # Get the date of the month in the desired format
        month_date_str = month_start_date.strftime('%B %Y')

        # Prepend the prefix "The Month of" and append the suffix ":"
        month_date_str = "The Month of " + month_date_str + ':'

        # Prepend the date and two newline characters to the summary
        summary = month_date_str + '\n\n' + summary

        # Append the new summary to the CSV file
        with open('monthly_summaries.csv', 'a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([month_start_date.strftime('%B %Y'), summary])

        print("---\n\n")
        print(summary)

def construct_daily_training_data():
    # This function constructs training data for GPT from multiple sources of summaries and thoughts.
    # It preprocesses the data to create "prompts" and "completions" for training GPT models.

    # Load the daily_summaries.csv file into a DataFrame
    daily_summaries_df = pd.read_csv('daily_summaries.csv', parse_dates=['Date'])

    # Define an array of prompt questions
    prompt_questions = [
        "What did John Ash think about on ",
        "Where was John's focus on ",
        "What were John Ash's thoughts on ",
        "What topics did John discuss on ",
        "What was on John Ash's mind on ",
        "What was John's primary concern on ",
        "Summarize John's thoughts from ",
        "What was the key focus of John Ash's thoughts on ",
        "What did John Ash reflect on during ",
        "What ideas did John explore on ",
        "Summarize John Ash's stream of thought on ",
        "What subjects did John Ash consider on ",
        "What was John Ash's cognitive focus on ",
        "What did John Ash write about on ",
        "What were John Ash's key insights on ",
        "Tell the story of John's thoughts on ",
        "What were John's observations on ",
        "What concepts did John Ash explore on ",
        "Summarize John's day on ",
        "What was John Ash's line of thought on "
    ]
    
    # Define an array of date formats
    date_formats = [
        "%B %d, %Y",  # January 01, 2022
        "%m/%d/%Y",   # 01/01/2022
        "%d %b %Y",   # 01 Jan 2022
        "%A, %B %d",  # Sunday, January 01
        "%d %B %Y",   # 01 January 2022
        "%d-%m-%Y",   # 01-01-2022
        "%Y-%m-%d",   # 2022-01-01
        "%b %d, %Y",  # Jan 01, 2022
        "%d %b, %Y",  # 01 Jan, 2022
        "%A, %d %B"   # Sunday, 01 January
    ]

    # Create an empty list to store the final prompts
    final_prompts = []

    # Construct prompts and completions
    for _, row in daily_summaries_df.iterrows():

        date = row['Date']
        summary = row['Summary']
        
        # Randomly select a prompt question and a date format
        prompt_question = random.choice(prompt_questions)
        date_format = random.choice(date_formats)
        
        # Format the date using the selected date format
        formatted_date = date.strftime(date_format)
        
        # Create the final prompt by concatenating the prompt question and the formatted date
        final_prompt = prompt_question + formatted_date
        
        # Append the final prompt and the corresponding summary (completion) to the final_prompts list
        final_prompts.append((final_prompt, summary))

    # Open the output file in write mode
    with open("temporal_iris.csv", 'w', newline='', encoding='utf-8') as csvfile:

        # Create a CSV writer object
        csv_writer = csv.writer(csvfile)
        
        # Write the header row to the CSV file
        csv_writer.writerow(['Prompt', 'Completion'])
        
        # Write each prompt and completion pair to the CSV file
        for prompt, completion in final_prompts:
            csv_writer.writerow([prompt, completion])

def construct_weekly_training_data():
    # Load weekly_summaries.csv into a DataFrame
    weekly_summaries_df = pd.read_csv('weekly_summaries.csv', parse_dates=['Week_Start_Date'])
    
    # Define an array of question prompts for weekly summaries
    prompt_questions = [
        "What topics did John discuss on the {week_ordinal} week of {month_name}?",
        "What was John Ash thinking about the {week_ordinal} week of {month_name}?",
        "Please summarize the {week_ordinal} week of {month_name}",
        "What notable occurred in John's life during the {week_ordinal} week of {month_name}?",
        "What did John Ash think about in the {week_ordinal} week of {month_name}?",
        "What thoughts did John discuss during the week ending on ",
        "Can you please summarize John's thoughts from the week ending ",
        "What did John Ash think about during the week of ",
        "Where was John's focus during the week ending on ",
        "What were John Ash's thoughts during the week of ",
        "What topics did John discuss in the week ending on ",
        "What was on John Ash's mind in the week ending on "
    ]
    
    # Define an array of date formats for weekly summaries
    date_formats = [
        "%B %d, %Y",  # January 01, 2022
        "%d %b %Y",   # 01 Jan 2022
        "%d/%m/%Y",   # 01/01/2022
    ]
    
    # Define a function to convert week number to ordinal string (e.g., 1 -> 1st, 2 -> 2nd)
    def ordinal(n):
        return str(n) + ("th" if 4 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th"))
    
    # Initialize an empty list to store the final prompts and completions
    final_prompts = []
    
    # Construct Prompts and Completions
    for _, row in weekly_summaries_df.iterrows():
        week_start_date = row['Week_Start_Date']
        summary = row['Summary']
        
        # Calculate the week number within the month
        week_number_in_month = (week_start_date.day - 1) // 7 + 1
        week_ordinal = ordinal(week_number_in_month)
        month_name = week_start_date.strftime('%B')
        
        # Randomly select a prompt question
        prompt_question = random.choice(prompt_questions)
        
        # Check if the prompt question contains placeholders for week ordinal and month name
        if '{week_ordinal}' in prompt_question and '{month_name}' in prompt_question:
            final_prompt = prompt_question.format(week_ordinal=week_ordinal, month_name=month_name)
        else:
            # Randomly select a date format
            date_format = random.choice(date_formats)
            # Format the date using the selected date format
            formatted_date = week_start_date.strftime(date_format)
            # Create the final prompt by concatenating the prompt question and the formatted date
            final_prompt = prompt_question + formatted_date
        
        # Append the final prompt and the corresponding summary (completion) to the final_prompts list
        final_prompts.append((final_prompt, summary))

    # Open the output file in write mode
    with open("weekly_iris.csv", 'w', newline='', encoding='utf-8') as csvfile:

        # Create a CSV writer object
        csv_writer = csv.writer(csvfile)
        
        # Write the header row to the CSV file
        csv_writer.writerow(['Prompt', 'Completion'])
        
        # Write each prompt and completion pair to the CSV file
        for prompt, completion in final_prompts:
            csv_writer.writerow([prompt, completion])

    
    return final_prompts

def construct_monthly_training_data():
    # Load the monthly_summaries.csv file into a DataFrame
    monthly_summaries_df = pd.read_csv('monthly_summaries.csv', parse_dates=['Month_Year'])
    
    # Define an array of question prompts for monthly summaries
    prompt_questions = [
        "What did John Ash think about during the month of ",
        "Where was John's focus in ",
        "What were John Ash's thoughts in ",
        "What topics did John discuss during ",
        "What was on John Ash's mind in ",
        "What was John's primary concern in ",
        "What was the key focus of John Ash's thoughts in ",
        "What did John Ash reflect on during ",
        "What ideas did John explore in ",
        "What subjects did John Ash consider in ",
    ]
    
    # Define an array of date formats for monthly summaries
    date_formats = [
        "%B %Y",
        "%B of %Y"
    ]
    
    # Initialize an empty list to store the final prompts and completions
    final_prompts = []
    
    # Construct Prompts and Completions
    for _, row in monthly_summaries_df.iterrows():
        month_year = row['Month_Year']
        summary = row['Summary']
        
        # Randomly select a prompt question and a date format
        prompt_question = random.choice(prompt_questions)
        date_format = random.choice(date_formats)
        
        # Format the date using the selected date format
        formatted_date = month_year.strftime(date_format)
        
        # Create the final prompt by concatenating the prompt question and the formatted date
        final_prompt = prompt_question + formatted_date
        
        # Append the final prompt and the corresponding summary (completion) to the final_prompts list
        final_prompts.append((final_prompt, summary))
    
    # Open the output file in write mode
    with open("monthly_iris.csv", 'w', newline='', encoding='utf-8') as csvfile:

        # Create a CSV writer object
        csv_writer = csv.writer(csvfile)
        
        # Write the header row to the CSV file
        csv_writer.writerow(['Prompt', 'Completion'])
        
        # Write each prompt and completion pair to the CSV file
        for prompt, completion in final_prompts:
            csv_writer.writerow([prompt, completion])
    
    return final_prompts

def create_seasonal_summaries():
    # This function will aggregate and summarize the monthly summaries for each season (winter, spring, summer, fall),
    # providing an overview of the key trends and patterns observed during that period.

    # Load monthly_summaries.csv into a DataFrame
    monthly_summaries_df = pd.read_csv('monthly_summaries.csv', parse_dates=['Month_Year'])
    monthly_summaries_df.set_index('Month_Year', inplace=True)

    # Define custom date ranges for each season based on approximate equinoxes and solstices
    seasons = {
        'Winter': ('12-21', '03-20'),
        'Spring': ('03-21', '06-20'),
        'Summer': ('06-21', '09-22'),
        'Fall': ('09-23', '12-20')
    }

    # Loop through each season
    for season, (start_day, end_day) in seasons.items():
        # Create a boolean mask to select rows within the date range for each season
        mask = ((monthly_summaries_df.index.month == int(start_day.split('-')[0])) & (monthly_summaries_df.index.day >= int(start_day.split('-')[1]))) | \
               ((monthly_summaries_df.index.month == int(end_day.split('-')[0])) & (monthly_summaries_df.index.day <= int(end_day.split('-')[1])))

        # Apply the mask to the DataFrame to get the data for the current season
        season_data = monthly_summaries_df[mask]

        # Generate summary for the current season
        season_text = f"Season of {season}:\n\n" + '\n\n'.join(season_data['Summary'].tolist())

        # Generate conversation and summary for the season
        conversation = [
            {"role": "system", "content": "Summarize John Ash's key thoughts and their connection to relevant world events for the season of " + season + ". Form a central focus and narrative through time rather than just summarizing a list of things he thought about. Mention how his thoughts might relate to world events."},
            {"role": "user", "content": season_text}
        ]

        summary = create_summary(conversation)

        print("---\n\n")
        print(summary)

def create_yearly_summaries(df):
    # This function will aggregate and summarize the monthly or seasonal summaries for each year,
    # capturing the most significant themes and learnings for the year.
    pass

def create_certainty_summaries(df):
    # This function will calculate the average certainty for different time periods
    # (e.g., daily, weekly, monthly) and provide summaries of the general level of certainty
    # for each period.
    pass

def create_sentiment_summaries(df):
    # This function will calculate the average sentiment (valence) for different time periods
    # and provide summaries of the general sentiment (positive or negative) for each period.
    pass

def create_temporal_focus_summaries(df):
    # This function will group thoughts by categories such as thought type (Reflect, Ask, Predict, State)
    # and calculate the average temporal focus (past, present, future) for each category over
    # a specified time period.
    pass

def create_trend_analysis(df):
    # This function will identify recurring themes, topics, or patterns over time and provide
    # summaries of notable trends observed in the data. It may use word counts.
    pass

def create_periodic_reflections(df):
    # This function will summarize key learnings, reflections, and takeaways for specific random time periods
    pass

def create_predictions_review(df):
    # This function will summarize predictions made during a specific time period and, if possible,
    # provide an evaluation of their accuracy based on subsequent outcomes.
    pass

def create_trackable_summaries(df):
    # This function will extract and organize trackable data (numeric variables that change over time)
    # from the text using a consistent format (e.g., "#weight: 175 lbs"). It will calculate summary
    # statistics for each trackable and generate summaries that describe the changes in trackables
    # over time, including any notable trends, patterns, or associations with other events.
    pass

# Load and preprocess the CSV into a DataFrame
df = load_and_preprocess_csv('prophet_thought_dump_ALL_THOUGHTS_2023.csv')

# Load fourthought labeled thoughts
df_thoughts = load_and_preprocess_csv('prophet_thought_dump_ALL_THOUGHTS_2023.csv')
df_thoughts['Platform'] = 'fourthought'

df_tweets = pd.read_csv('twitter_archive.csv')
df_tweets['Platform'] = 'twitter'

# Rename 'created_at' column to 'Post date' in df_tweets to match df_thoughts
df_tweets.rename(columns={'created_at': 'Post date'}, inplace=True)

# Drop the 'lang' column from df_tweets
df_tweets.drop(columns=['lang'], inplace=True)

# Convert 'Post date' column to datetime in df_tweets
df_tweets['Post date'] = pd.to_datetime(df_tweets['Post date'], format='%a %b %d %H:%M:%S %z %Y').dt.tz_convert('US/Pacific')
df_tweets['retweet_count'] = df_tweets['retweet_count'].astype("Int64")
df_tweets['favorite_count'] = df_tweets['favorite_count'].astype("Int64")

# Merge the two dataframes based on 'Post date'
df_merged = pd.concat([df_thoughts, df_tweets], axis=0, ignore_index=True, sort=False)

# Sort the merged dataframe by 'Post date'
df_merged.sort_values(by='Post date', inplace=True)
df_merged.reset_index(drop=True, inplace=True)

# Generate daily summaries using the preprocessed DataFrame
#daily_summaries = create_daily_summaries(df_merged)
#weekly_summaries = create_weekly_summaries()
#monthly_summaries = create_monthly_summaries()
#construct_weekly_training_data()
construct_monthly_training_data()