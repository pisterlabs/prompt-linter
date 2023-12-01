import cohere
import configparser
from cohere.responses.classify import Example

config_read = configparser.ConfigParser()
config_read.read("config.ini")

basketball_teams = ['Celtics',
                    'Nets',
                    'Knicks',
                    '76ers',
                    'Raptors',
                    'Warriors',
                    'Clippers',
                    'Lakers',
                    'Suns',
                    'Kings',
                    'Bulls',
                    'Cavaliers',
                    'Pistons',
                    'Pacers',
                    'Bucks',
                    'Hawks',
                    'Hornets',
                    'Heat',
                    'Magic',
                    'Wizards',
                    'Nuggets',
                    'Timberwolves',
                    'Thunder',
                    'Blazers',
                    'Jazz',
                    'Mavericks',
                    'Rockets',
                    'Grizzlies',
                    'Pelicans',
                    'Spurs']

moneyline_examples = [
    Example("I want to place a moneyline on the Raptors", "Moneyline"),
    Example("I want to bet $6 on Wizards, moneyline", "Moneyline"),
    Example("Thunder moneyline twenty dollars", "Moneyline"),
    Example("Bet on Warriors winning", "Moneyline"),
    Example("Lakers win", "Moneyline"),
    Example("Win for raptors", "Moneyline"),
    Example("Can I bet on a win for my team", "Moneyline"),
    Example("Jazz winning the game", "Moneyline"),
    Example("Celtics victory", "Moneyline"),
    Example("I wanna put money on the Bucks getting the victory", "Moneyline"),
    Example("Parlay on Pelicans victory", "Moneyline"),

    Example("I'd like to place a bet on the Lakers", "N/A"),
    Example("Place a bet on the Toronto Maple Leafs.", "N/A"),
    Example("I'm just not using key words here", "N/A"),
    Example("You get the idea", "N/A"),
    Example("Place a wager on my team", "N/A"),
    Example("Parlay on Cavaliers", "N/A"),
    Example("Place a bet on the Golden State Warriors", "N/A"),
    Example("Place a over/under bet", "N/A"),
    Example("Celtics will score over 16 points", "N/A"),
    Example("I want to place a bet on Steph Curry having 12 points.", "N/A"),
    Example("Bet on 42 points for the Green Bay Packers", "N/A"),
    Example("Place a moneyline bet on Steelers getting 7 points", "N/A")
]


def get_sport(prompt):
    api_key = config_read.get("api_keys", "sports")
    co = cohere.Client(api_key)
    pre_prompt = "If a sport is referenced in the following prompt, please output the name of the sport and nothing else. Otherwise, say 'N/A'. Here is the prompt: "
    response = co.generate(
        pre_prompt + prompt,
        max_tokens=20
    )
    sport = response[0].text.strip()
    if sport[:1] == '\n':
        sport = sport[1:]
    if "N/A" in sport:
        return None
    return sport


def get_bet_amount(prompt):
    api_key = config_read.get("api_keys", "prices")
    co = cohere.Client(api_key)
    model = config_read.get("models", "prices")
    response = co.generate(
        prompt,
        model=model,
        max_tokens=20
    )
    bet = response[0].text
    if bet == 'N/A':
        return None
    try:
        bet = float(bet)
        return bet
    except ValueError:
        return None


def get_team(prompt):
    api_key = config_read.get("api_keys", "teams")
    co = cohere.Client(api_key)
    model = config_read.get("models", "teams")
    response = co.generate(
        prompt,
        model=model,
        max_tokens=20
    )
    team = response[0].text
    if team == 'N/A':
        return None
    for t in basketball_teams:
        if t.lower() in team.lower():
            team = t
    if team not in basketball_teams:
        return None
    return team


def get_points(prompt):
    api_key = config_read.get("api_keys", "points")
    co = cohere.Client(api_key)
    model = config_read.get("models", "points")
    response = co.generate(
        prompt,
        max_tokens=20,
        model=model
    )
    points = response[0].text
    if points == "N/A":
        return None
    if not points.isdigit():
        return None
    return int(points)


def get_win(prompt):
    api_key = config_read.get("api_keys", "win")
    co = cohere.Client(api_key)
    response = co.classify(
        inputs=[prompt],
        examples=moneyline_examples,
    )
    if response[0].confidence < 0.60 or response[0].prediction == "N/A":
        return None
    return True


# print([get_team("the Raptors")])
