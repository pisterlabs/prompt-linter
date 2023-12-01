# %%
import os
import openai

# %%
%load_ext jupyter_ai_magics

# %%
%ai help

# %%
%ai list

# %%
def celsius_to_fahrenheit(celsius):
    return (celsius * 9/5) + 32

# %%
def celsius_to_fahrenheit(celsius):
    return (celsius * 9/5) + 32

celsius_to_fahrenheit(20)


# %%
celsius_to_fahrenheit(0)

# %%
celsius_to_fahrenheit(100)

# %%
%%ai chatgpt -f code

a function that returns its square root.

# %%
import math
def square_root(x):
    return math.sqrt(x)

# %%
import math
def square_root(x):
    return math.sqrt(x)

# %%
def square_root(x):
    return x**0.5

# %%
import math

def square_root(x):
    return math.sqrt(x)

# %%
def square_root(x):
    return x**(0.5)

# %%
import math

def square_root(x):
    if x < 0:
        raise ValueError("Input must be a non-negative number")
    try:
        return math.sqrt(x)
    except ValueError as e:
        print(f"Error calculating square root: {e}")

# Example usage
try:
    result = square_root(16)
    print(result)
except ValueError as e:
    print(e)

    # return math.sqrt(x)

# %%
def square_root(x):
    return x**0.5

# Example usage:
print(square_root(9))

# %%
import math
def get_square_root(num):
    return math.sqrt(num)

# %%
import math

def square_root(n):
    return math.sqrt(n)

# %%
# Here is a simple Python function that returns the square root of a given number:


import math

def square_root(number):
    return math.sqrt(number)


# Note: This answer assumes that you are looking for a programming code snippet and not the mathematical formula for square roots.

# %%
import math

def square_root(input):
    return math.sqrt(input)

# %%
import math
def square_root(num):
    return math.sqrt(num)

# %%
import math
def square_root(x):
    return math.sqrt(x)

# %%
def square_root(x):
    return x ** (1 / 2)

# %%
def square_root(x):
    return x ** 0.5

# %%
import math
def square_root(x):
    return math.sqrt(x)

# %%
%%ai openai-chat:gpt-3.5-turbo -f code

A function that returns the cube root of a number

# %%
def cube_root(x):
    return x**(1/3)

# %%
def cube_root(x):
    return x**(1/3)

# %%
def cube_root(x):
    return x**(1/3)

# %%
def cube_root(x):
    return x**(1/3)

# %%
import math
def get_cube_root(num):
    return math.pow(num, 1/3)

# %%
def get_cube_root(num):
    return num**(1/3)

# %%
# Certainly! Here's a Python function that returns the cube root of a given number:


def cube_root(number):
    return number ** (1/3)


# Note: This code assumes that the number provided as an argument is a positive real number.

# %%
import math

def cube_root(input):
    return input ** (1/3)

# %%
import math
def cube_root(num):
    return num**(1/3)

# %%
def cube_root(x):
    return x ** (1/3)

# %%
def cube_root(x):
    return x ** (1 / 3)

# %%
import requests

# Make a GET request to an API endpoint
response = requests.get('https://nba-stats-db.herokuapp.com/api/playerdata/name/Jayson Tatum')

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Access the response data
    data = response.json()
    print(data)
else:
    print('Error:', response.status_code)

# %%
%%ai openai-chat:gpt-3.5-turbo -f code

Sort the data from above to find Jayson Tatums highest scoring season.

# %%
data = {
    'Jayson Tatum': [
        {"Season": "2017-2018", "Points": 13.9},
        {"Season": "2018-2019", "Points": 15.7},
        {"Season": "2019-2020", "Points": 23.4},
        {"Season": "2020-2021", "Points": 26.4}
    ]
}

highest_season = max(data['Jayson Tatum'], key=lambda x: x['Points'])
highest_season['Season']

# %%
data = {
    'Jayson Tatum': {
        '2017-2018': 13.9,
        '2018-2019': 15.7,
        '2019-2020': 23.4,
        '2020-2021': 26.4
    }
}

highest_scoring_season = max(data['Jayson Tatum'], key=lambda x: data['Jayson Tatum'][x])

highest_scoring_season

# %%
tatum_stats = [
    {"season": "2017-2018", "points": 1399},
    {"season": "2018-2019", "points": 1533},
    {"season": "2019-2020", "points": 1545},
    {"season": "2020-2021", "points": 1573}
]

sorted_stats = sorted(tatum_stats, key=lambda x: x["points"], reverse=True)
highest_scoring_season = sorted_stats[0]["season"]

# %%
# List of Jayson Tatum's seasons and their corresponding scores
seasons = [
    {"season": "2017-2018", "score": 13.9},
    {"season": "2018-2019", "score": 15.7},
    {"season": "2019-2020", "score": 23.4},
    {"season": "2020-2021", "score": 26.4}
]

# Sorting the seasons based on score in descending order
sorted_seasons = sorted(seasons, key=lambda x: x["score"], reverse=True)

# Jayson Tatum's highest scoring season
highest_scoring_season = sorted_seasons[0]["season"]

# %%
player_stats = {
    "Jayson Tatum": [15, 18, 21, 23, 20]
}

highest_scoring_season = max(player_stats["Jayson Tatum"])
highest_scoring_season


