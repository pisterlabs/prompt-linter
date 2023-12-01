import csv
import os
import shutil
import openai
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
import re
import math
from halo import Halo
from faker import Faker
import random
import numpy as np

fake = Faker()
openai.api_key = config.OPENAI_API_KEY
spinner = Halo(text='Loading', spinner='dots')

def chat_gpt_completion(chat_message, append = False, usr_prompt = ""):

    spinner.start("text generating")

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages= chat_message
    )

    spinner.succeed("text generated")

    response = completion['choices'][0]["message"]["content"]

    if append:
        chat_message.append({"role": "system", "content": response})

    return response, chat_message

def write_review(movie_title, movie_year, directors, rate):
    prompt = f"""
    Use a temperature of 0.8, Write a short (less than 100 words) funny review for the movie
    {movie_title} ({movie_year}), directed by {directors}
    Your rating of the movie out of 5 is {rate}/5 so write the review accordingly you don't have to mention your rating.
    """

    review, _ = chat_gpt_completion([{"role": "user", "content": prompt}])



    return review


def write_review_csv(filename, movie_title, movie_year, directors, rating, watched_date, review):
    # Check if the file exists, if not create it and add the headers
    if not os.path.isfile(filename):
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Title', 'Year', 'Directors', 'Rating', 'WatchedDate','Review']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    # Append the review data to the CSV file
    with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Title', 'Year', 'Directors', 'Rating', 'WatchedDate', 'Review']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({
            'Title': movie_title,
            'Year': movie_year,
            'Directors': directors,
            'Rating': rating,
            'WatchedDate': watched_date,
            'Review': review
        })


def get_top_movies(stop, file_name = "250Movies.csv"):
    with open(file_name, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)

        # Skipping the header row
        header = next(csvreader)
        # random.shuffle(csvreader)

        # Processing the rows
        list_movies = []
        for i, row in enumerate (csvreader):
            rank, name, year, rating, *_, directors = row[:12]
            movie_thing = [name, year, directors, rating]
            print(f"Title: {name}, Year: {year}, Directors: {directors}, Rating: {rating}")
            list_movies.append(movie_thing)
            if i == stop:
                break

        return list_movies

def calculate_rate(rating):
    # Generate a random number from a normal distribution with a mean of 0 and a standard deviation of 1
    random_number = np.random.normal(0, 1)

    # Round the random number to the nearest integer
    random_integer = int(round(random_number))

    # Limit the random_integer to the range of -3 to 3
    random_integer = max(-3, min(3, random_integer))

    rate = math.floor(float(rating)) / 2 + random_integer

    # Clamp the rate between 0 and 5
    rate = max(0, min(5, rate))

    return str(rate)



def writeTopMovies(num):

    thing = get_top_movies(num)

    for x in thing:
        movie_title = x[0]
        year = x[1]
        directors = x[2]
        rating = x[3]
        rate = calculate_rate(rating)
        review = write_review(movie_title, year, directors, rate)
        date = fake.date_between(start_date='-5y', end_date='today')
        date = date.strftime("%Y-%m-%d")
        write_review_csv("letterboxd_reviews.csv", movie_title, year, directors, rate, date, review)


def shuffle_csv (file_name):
    with open(file_name, 'r', encoding='utf-8') as infile:
        csvreader = csv.reader(infile)

        # Read the header row
        header = next(csvreader)

        # Read the rest of the rows
        rows = list(csvreader)

    # Shuffle the rows
    random.shuffle(rows)

    # Write the shuffled rows to a new CSV file
    with open(file_name, 'w', encoding='utf-8', newline='') as outfile:
        csvwriter = csv.writer(outfile)

        # Write the header row
        csvwriter.writerow(header)

        # Write the shuffled rows
        for row in rows:
            csvwriter.writerow(row)


def generate_movie_reviews(num_reviews):
    source_file = 'letterboxd_reviews.csv'
    destination_folder = 'files/letterboxd_reviews.csv'  # Replace with the actual path to the destination folder

    shuffle_csv("250Movies.csv")
    writeTopMovies(num_reviews)

    # Check if the source file exists
    if os.path.exists(source_file):
        # Check if the destination file exists
        destination_file = os.path.join(destination_folder, os.path.basename(source_file))
        if os.path.exists(destination_file):
            # Remove the existing destination file
            os.remove(destination_file)

        # Move the source file to the destination folder
        shutil.move(source_file, destination_folder)
        print("File moved successfully.")
    else:
        print("Source file not found.")

