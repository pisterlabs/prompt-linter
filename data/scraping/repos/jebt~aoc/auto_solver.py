import json
import os
import sys
import time
import zoneinfo
from datetime import datetime
from openai import OpenAI

import requests


# entry function that checks the time and starts the thing or counts down until the problem is available, checking like
#   from 1 second before the problem should release, keep retrying with some time in between until it's available or
#   until some max number of retries is reached
# get problem description
# make the api call to openai to get the python code (the call should have examples (2015-2022 day 1 at least))
# get input (option to save to disk but just in memory for speed)
# run the code (with eval?) on the input and get the answer
# submit the answer to part 1
# open a browser with the scoreboard

# fetch part 2 (a few retries with a little time in between if it fails the first time)
# make the api call to openai to get the python code (the call should have examples)
# run the code (with eval?) on the input and get the answer
# submit the answer to part 2
# open a browser with the scoreboard

class AutoSolver:
    def __init__(self, year, day):
        self.prevent_calls_to_aoc_website = True
        with open("auto_solver/gpt_messages.json", 'r') as file:
            self.gpt_messages: list = json.load(file)
        self.year = year
        self.day = day
        self.api_key = os.environ["OPENAI_API_KEY"]
        self.cookie_header = os.environ["AOC_COOKIE_HEADER"]
        self.cookie_header_test = os.environ["AOC_COOKIE_HEADER_TEST"]
        self.client = OpenAI()
        self.description = None
        self.code = None
        self.input = None
        self.answer1 = None
        self.answer2 = None

    # entry function that checks the time and starts the thing or counts down until the problem is available
    def arm(self):
        # get the current time and check if the problem should be released yet
        current_time = datetime.now().astimezone()
        puzzle_release = datetime(year=self.year, month=12, day=self.day, hour=6,
                                  tzinfo=zoneinfo.ZoneInfo(key="Europe/Amsterdam"))
        seconds_to_release = (puzzle_release - current_time).total_seconds()
        if seconds_to_release > 0:
            print(f"Problem should be released in {seconds_to_release} seconds.")
            # if it's less than an hour, start a countdown, if it's more than an hour quit the program
            if seconds_to_release > 3600:
                raise Exception("It should be longer than an hour from now before this puzzle will be released, "
                                f"try again later. {seconds_to_release=}")
            else:
                wait_input = input(f"Puzzle should be released in {seconds_to_release} seconds. Keep the program "
                                   f"running to auto-fetch input as soon as possible? (y/n): ")
                if wait_input.lower() in ["y", "yes"]:
                    raise NotImplementedError("arm")  # todo: implement
                else:
                    print("Shutting down.")
                    sys.exit()
        else:
            print("Problem should be released already.")
            # start a countdown of at least 10 seconds until a round minute
            seconds_until_round_minute = 60 - current_time.second
            if seconds_until_round_minute < 10:
                seconds_until_round_minute += 60
            print(f"Waiting {seconds_until_round_minute} seconds until a round minute.")

            self.count_down(seconds_until_round_minute)
            self.fetch_problem_description()
            final_gpt_message = {
                "role": "user",
                "content": f"{self.description}"
            }
            self.gpt_messages += [final_gpt_message]
            self.get_code_from_openai()
            self.fetch_input()
            self.answer1 = self.get_answer_with_exec()
            print(f"Answer 1: {self.answer1}")
            # todo: implement

    def fetch_and_save_descriptions(self):
        years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
        for year in years:
            day = 1
            url = f"https://adventofcode.com/{year}/day/{day}"
            headers = {
                "Cookie": self.cookie_header_test
            }

            response = requests.get(url, headers=headers)
            response.raise_for_status()
            page = response.text
            # get the part between the <main> tags
            description = page.split("<main>")[1].split("</main>")[0]

            # save the description to a file
            with open(f"auto_solver/part1_descriptions/{year}_{day}_part1.html", "w") as f:
                f.write(description)

    def get_answer_with_exec(self):
        context = {}
        exec(self.code, context)
        solve = context["solve"]
        answer = solve(self.input)
        return answer

    def get_code_from_openai(self):
        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=self.gpt_messages
        )
        print(completion.choices[0].message.content)
        with open(f"auto_solver/gpt_output.txt", "w") as f:
            f.write(completion.choices[0].message.content)
        with open(f"auto_solver/gpt_output_object.txt", "w") as f:
            f.write(str(completion.choices[0].message))

        self.code = completion.choices[0].message.content

    def fetch_input(self):
        if self.prevent_calls_to_aoc_website:
            with open(f"year_{self.year}/day_{self.day:02}_input.txt", "r") as f:
                input_data = f.read()
        else:
            url = f"https://adventofcode.com/{self.year}/day/{self.day}/input"
            headers = {
                "Cookie": self.cookie_header_test
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            input_data = response.text
            with open(f"auto_solver/fetched_input.txt", "w") as f:
                f.write(input_data)
        self.input = input_data

    def fetch_problem_description(self):
        if self.prevent_calls_to_aoc_website:
            with open(f"auto_solver/part1_descriptions/{self.year}_{self.day}_part1.html", "r") as f:
                description = f.read()
        else:
            url = f"https://adventofcode.com/{self.year}/day/{self.day}"
            headers = {
                "Cookie": self.cookie_header_test
            }

            response = requests.get(url, headers=headers)
            response.raise_for_status()
            page = response.text
            description = page.split("<main>")[1].split("</main>")[0]
            # save to a file
            with open(f"auto_solver/gpt_food.html", "w") as f:
                f.write(description)

        self.description = description

    @staticmethod
    def count_down(seconds):
        testing = False
        if testing:
            seconds = 3
            for i in range(seconds):
                print(f"Waiting. {seconds - i} seconds left until fetch...")
                time.sleep(1)
        else:
            time_left = seconds
            while time_left > 0:
                if time_left < 55:  # we want the extra precision and this doesn't mess with the atleast 10 seconds
                    current_time = datetime.now().astimezone()  # update the time left because sleep might not be accurate
                    seconds_until_round_minute = 60 - current_time.second
                    time_left = seconds_until_round_minute
                print(f"Waiting. {time_left} seconds left until fetch...")
                time.sleep(1)
                time_left -= 1
