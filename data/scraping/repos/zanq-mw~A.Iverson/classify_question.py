import cohere
from cohere.responses.classify import Example
import configparser

config_read = configparser.ConfigParser()
config_read.read("config.ini")

co = cohere.Client(config_read.get("api_keys", "classify_question"))

examples = [
    Example("What is a moneyline?", "Question"),
    Example("How do I place a bet?", "Question"),
    Example("What is a quick bet?", "Question"),
    Example("What is a teaser bet?", "Question"),
    Example("What are alternate lines?", "Question"),
    Example("Are there different types of odds?", "Question"),
    Example("Can I bet during a match?", "Question"),
    Example("Can I cancel my bet?", "Question"),
    Example("What are the types of bets?", "Question"),
    Example("What events can I bet on?", "Question"),
    Example("What is a parlay?", "Question"),
    Example("How much money will I make?", "Question"),
    Example("How do I make my selections", "Question"),
    Example("How much money will I make?", "Question"),
    Example("Can you explain what a parlay bet is", "Question"),
    Example("How much money will I make?", "Question"),
    Example("I don't understand what a teaser bet is", "Question"),

    Example("I'd like to place a bet on the Lakers winning.", "Bet"),
    Example("Place a bet on the Toronto Maple Leafs losing 3-0.", "Bet"),
    Example("Parlay bet of Houston Texans over 38.5 total and NY Jets winning", "Bet"),
    Example("I would like to place a teaser bet on New York Giants winning and Minnesota Vikings winning.", "Bet"),
    Example("Bet on Joe Burrow over 34.5 total points", "Bet"),
    Example("Parlay of Cavaliers losing and Raptors winning", "Bet"),
    Example("Place a bet on the Golden State Warriors", "Bet"),
    Example("Place a teaser bet", "Bet"),
    Example("Can you place a teaser bet of the Sens and Canadians losing.", "Bet"),
    Example("I want to place a bet on Steph Curry.", "Bet"),
    Example("Bet on the Green Bay Packers", "Bet"),
    Example("Place a moneyline and match spread bet on the Steelers", "Bet"),
    Example("Can I put money on the golden state warriors", "Bet"),

    Example("Place a fucking bet you piece of shit", "Inappropriate"),
    Example("motherfucker my team lost", "Inappropriate"),
    Example("goddammit fucking shit", "Inappropriate"),
    Example("AIverson you cunt", "Inappropriate"),
    Example("You bitches", "Inappropriate"),
]


def bet_or_question(prompt):
    response = co.classify(
        inputs=[prompt],
        examples=examples,
    )
    if response[0].confidence < 0.80 or response[0].prediction == "Inappropriate":
        response = "I do not understand your request. Please be more specific."
    else:
        response = response[0].prediction

    return response
