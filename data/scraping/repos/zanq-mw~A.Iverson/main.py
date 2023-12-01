import cohere
from copy import deepcopy

import api
import bet_attributes
import classify_question
import configparser
import questions

from api_responses import Bet, Mode

config_read = configparser.ConfigParser()
config_read.read("config.ini")
api_key = config_read.get("api_keys", "generate_answers")
co = cohere.Client(api_key)

GAME_DATA = [
    {"home": "Raptors", "away": "Clippers",
        "name": "Raptors vs Clippers", "multiplier": 1.7, "odds": "+157"},
    {"home": "Raptors", "away": "Mavericks",
        "name": "Raptors vs Mavericks", "multiplier": 1.2, "odds": "-157"},
    {"home": "Hawks", "away": "Raptors",
        "name": "Hawks vs Raptors", "multiplier": 1.5, "odds": "+198"},
    {"home": "Lakers", "away": "Clippers",
        "name": "Lakers vs Clippers", "multiplier": 1.3, "odds": "-198"},
    {"home": "Lakers", "away": "Mavericks",
        "name": "Lakers vs Mavericks", "multiplier": 1.1, "odds": "+62"},
    {"home": "Hawks", "away": "Lakers", "name": "Hawks vs Lakers",
        "multiplier": 1.8, "odds": "-62"},
    {"home": "Warriors", "away": "Clippers",
        "name": "Warriors vs Clippers", "multiplier": 1.9, "odds": "+81"},
    {"home": "Warriors", "away": "Mavericks",
        "name": "Warriors vs Mavericks", "multiplier": 1.6, "odds": "-17"},
    {"home": "Hawks", "away": "Warriors",
        "name": "Hawks vs Warriors", "multiplier": 1.4, "odds": "+101"},
    {"home": "Celtics", "away": "Spurs",
        "name": "Celtics vs Spurs", "multiplier": 1.7, "odds": "+157"},
    {"home": "Celtics", "away": "Pelicans",
        "name": "Celtics vs Pelicans", "multiplier": 1.2, "odds": "-157"},
    {"home": "Thunder", "away": "Celtics",
        "name": "Thunder vs Celtics", "multiplier": 1.5, "odds": "+198"},
    {"home": "Nets", "away": "Spurs",
        "name": "Nets vs Spurs", "multiplier": 1.3, "odds": "-198"},
    {"home": "Nets", "away": "Pelicans",
        "name": "Nets vs Pelicans", "multiplier": 1.1, "odds": "+62"},
    {"home": "Thunder", "away": "Nets", "name": "Thunder vs Nets",
        "multiplier": 1.8, "odds": "-62"},
    {"home": "Cavaliers", "away": "Spurs",
        "name": "Cavaliers vs Spurs", "multiplier": 1.9, "odds": "+81"},
    {"home": "Cavaliers", "away": "Pelicans",
        "name": "Cavaliers vs Pelicans", "multiplier": 1.6, "odds": "-17"},
    {"home": "Thunder", "away": "Cavaliers",
        "name": "Thunder vs Cavaliers", "multiplier": 1.4, "odds": "+101"},
    {"home": "Knicks", "away": "76ers",
        "name": "Knicks vs 76ers", "multiplier": 1.3, "odds": "+120"},
    {"home": "Bulls", "away": "Pistons",
        "name": "Bulls vs Pistons", "multiplier": 1.5, "odds": "-150"},
    {"home": "Pacers", "away": "Bucks",
        "name": "Pacers vs Bucks", "multiplier": 1.6, "odds": "+140"},
    {"home": "Hornets", "away": "Raptors",
        "name": "Hornets vs Raptors", "multiplier": 1.2, "odds": "-120"},
    {"home": "Clippers", "away": "Heat",
        "name": "Clippers vs Heat", "multiplier": 1.8, "odds": "+180"},
    {"home": "Magic", "away": "Wizards",
        "name": "Magic vs Wizards", "multiplier": 1.7, "odds": "-170"},
    {"home": "Nuggets", "away": "Timberwolves",
        "name": "Nuggets vs Timberwolves", "multiplier": 1.4, "odds": "+157"},
    {"home": "Trail Blazers", "away": "Jazz",
        "name": "Trail Blazers vs Jazz", "multiplier": 1.5, "odds": "+150"},
    {"home": "Suns", "away": "Kings",
        "name": "Suns vs Kings", "multiplier": 1.9, "odds": "-198"},
    {"home": "Rockets", "away": "Grizzlies",
        "name": "Rockets vs Grizzlies", "multiplier": 1.1, "odds": "+110"}
]


def bet_workflow(prompt):
    data = {'sport': 'basketball',
            'team': bet_attributes.get_team(prompt),
            'bet_amount': bet_attributes.get_bet_amount(prompt),
            'points': bet_attributes.get_points(prompt)}
    return validate_bet_data(data)


def validate_bet_data(data):

    if 'sport' not in data or data['sport'] is None:
        return {
            "bet": None,
            "bot_message": "What sport would you like to place your bet on?",
            "mode": Mode.BET,
            "bet_data": data,
            "suggested_prompts": ["Basketball", "Soccer", "Hockey", "Football", "Baseball", "Ultimate Frisbee", "Exit"]
        }
    if 'team' not in data or data['team'] is None:
        return {
            "bet": None,
            "bot_message": "What team would you like to place your bet on?",
            "mode": Mode.BET,
            "bet_data": data,
            "suggested_prompts": ["Exit"]
        }
    elif 'game_title' not in data or data['game_title'] is None:
        games = []
        multipliers = []
        odds = []
        suggested_prompts = []
        msg = 'Here are the options:\n\n'
        for i in range(len(GAME_DATA)):
            game = GAME_DATA[i]
            if data['team'] in [game['home'], game['away']]:
                games.append(game['name'])
                multipliers.append(game['multiplier'])
                odds.append(game['odds'])
        for i in range(len(games)):
            suggested_prompts.append(str(i+1))
            game = games[i]
            msg += (f'{i+1}. {game}\n')
        suggested_prompts.append("Exit")
        return {
            "bet": None,
            "bot_message": msg[:len(msg)-1],
            "mode": Mode.BET,
            "bet_data": data,
            "suggested_prompts": suggested_prompts
        }
    elif ('points' not in data or data['points'] is None) and ('win' not in data or data['win'] is None):
        return {
            "bet": None,
            "bot_message": 'What outcome would you like to bet on?',
            "mode": Mode.BET,
            "bet_data": data,
            "suggested_prompts": ["Exit"]
        }
    elif 'bet_amount' not in data or data['bet_amount'] is None:
        return {
            "bet": None,
            "bot_message": 'How much would you like to bet?',
            "mode": Mode.BET,
            "bet_data": data,
            "suggested_prompts": ["$20", "$50", "$100", "$200", "Exit"]
        }
    else:
        if 'points' in data and data['points'] is not None:
            data['bet_title'] = f"Over {data['points'] - 0.5}"
            data['bet_description'] = 'Total'
        else:
            data['bet_title'] = data['team']
            data['bet_description'] = 'Moneyline'
        bet = Bet(
            game_title=data['game_title'],
            bet_title=data['bet_title'],
            bet_description=data['bet_description'],
            bet_amount=data['bet_amount'],
            to_win=data['bet_amount']*data['multiplier'],
            odds=data['odds']
        )

        return {
            "bet": bet,
            "bot_message": "I've put together a betslip for you, open it to place your bet.",
            "mode": Mode.NO_TYPE,
            "bet_data": None,
            "suggested_prompts": ['What is a straight bet?', 'What is moneyline?', 'How do I place a bet?']
        }


def add_to_bet_data(user_message, user_data):
    if user_message.lower() == "exit":
        return {
            "bot_message": "Okay, I've abandoned that bet. Is there anything else I can help you with?",
            "mode": Mode.NO_TYPE,
            "bet": None
        }

    data = {
        "sport": user_data.sport,
        "team": user_data.team,
        "bet_amount": user_data.bet_amount,
        "points": user_data.points,
        "game_title": user_data.game_title,
        "multiplier": user_data.multiplier,
        "odds": user_data.odds,
    }
    initial_data = deepcopy(data)

    if 'sport' not in data or data['sport'] is None:
        data['sport'] = bet_attributes.get_sport(user_message)
    elif 'team' not in data or data['team'] is None:
        data['team'] = bet_attributes.get_team(user_message)
    elif 'game_title' not in data or data['game_title'] is None:
        games = []
        multipliers = []
        odds = []

        for i in range(len(GAME_DATA)):
            game = GAME_DATA[i]
            if data['team'] in [game['home'], game['away']]:
                games.append(game['name'])
                multipliers.append(game['multiplier'])
                odds.append(game['odds'])

        if user_message.isdigit() and int(user_message) <= len(games):
            user_input = int(user_message) - 1
            data['game_title'] = games[user_input]
            data['multiplier'] = multipliers[user_input]
            data['odds'] = odds[user_input]
    elif ('points' not in data or data['points'] is None) and ('win' not in data or data['win'] is None):
        data['points'] = bet_attributes.get_points(user_message)
        data['win'] = bet_attributes.get_win(user_message)
    elif 'bet_amount' not in data or data['bet_amount'] is None:
        data['bet_amount'] = bet_attributes.get_bet_amount(user_message)

    validated = validate_bet_data(data)
    if validated['bet_data'] == initial_data and validated['bet'] is None:
        validated['bot_message'] = "Sorry, I didn't get that. Please try again."
    return validated


def question_workflow(prompt):
    response = questions.generate_answer(prompt)
    answer = response + "\n\nWas this helpful?"

    return {
        "bot_message": answer,
        "mode": Mode.QUESTION,
        "suggested_prompts": ['Yes', 'No'],
        "saved_question": [prompt, response]
    }


def start_workflow(user_input):
    if user_input.mode == Mode.BET.value:
        return add_to_bet_data(user_input.user_message, user_input.bet_data)
    elif user_input.mode == Mode.QUESTION.value:
        if user_input.user_message.lower() == "yes":
            data = f'"prompt": "{user_input.saved_question[0]}", "completion": "{user_input.saved_question[1]}"'
            data = '\n{' + data + '}'
            with open('training-data/generate_training_data.jsonl', "a") as f:
                f.write(data)

            msg = "Thanks for letting me know. Anything else I can help you with?"

        else:
            msg = "Sorry for that. Anything else I can help you with?"
        return {
            "bot_message": msg,
            "mode": Mode.NO_TYPE,
        }

    else:
        req_type = classify_question.bet_or_question(user_input.user_message)

    response = {}

    if req_type == "Bet":
        response = bet_workflow(user_input.user_message)
    elif req_type == "Question":
        return question_workflow(user_input.user_message)
    else:
        answer = "I do not understand, please try again."
        response = {
            "bot_message": answer,
            "mode": Mode.NO_TYPE,
            "bet": None
        }
    return response
