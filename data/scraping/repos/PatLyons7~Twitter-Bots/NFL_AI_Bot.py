# Import necessary libraries
import openai
import tweepy
import random

# access twitter and openAI API keys
fref = open('Desktop/Twitter-Bots/NFL_AI_Bot/keys.txt','r', newline = '\n')
key_string = fref.read()
fref.close()
keys = key_string.split("_")

CONSUMER_KEY = keys[0]
CONSUMER_SECRET = keys[1]
ACCESS_KEY = keys[2]
ACCESS_SECRET = keys[3]
BEARER_TOKEN = keys[4]
OPENAI_KEY = keys[5][:-1]

client = tweepy.Client(BEARER_TOKEN,CONSUMER_KEY,CONSUMER_SECRET,ACCESS_KEY, ACCESS_SECRET)
auth = tweepy.OAuth1UserHandler(CONSUMER_KEY, CONSUMER_SECRET,ACCESS_KEY, ACCESS_SECRET)
api = tweepy.API(auth)

# Set openai API key
openai.api_key = OPENAI_KEY

#List of NFL teams to choose from
teams = [ "NFL", "Tampa Bay Buccaneers", "Atlanta Falcons",
         "New Orleans Saints", "Washington Commanders", "New York Giants",
         "New York Jets", "New ENngland Patriots", "Buffalo Bills",
         "Dallas Cowboys", "Philadelphia Eagles", "Carolina Panthers",
         "Miami Dolphins", "Jacksonville Jaguars", "Indianapolis Colts",
         "Tennessee Titans", "Houston Texans", "Arizona Cardinals",
         "Los Angeles Rams", "Los Angeles Chargers", "Seattle Seahawks",
         "San Francisco 49ers", "Denver Broncos", "Kansas City Chiefs",
         "Las Vegas Raiders", "Chicago Bears", "Detroit Lions",
         "Green Bay Packers", "Minnesota Vikings", "Baltimore Ravens",
         "Cleveland Browns", "Pittsburgh Steelers", "Cincinatti Bengals"
]

#List of NFL Players to choose from
players = ["Tom Brady", "Zach Wilson", "Mike White", "Jeff Wilson",
           "CeeDee Lamb", "Josh Allen", "Jaxson de Ville",
           "Sir Purr", "Christian McCaffrey", "Saquon Barkley","Sam Darnold",
           'Justin Herbert', 'Patrick Mahomes', "Rob Gronkowski",
           'Matthew Stafford', 'Aaron Rodgers', 'Dak Prescott', 'Joe Burrow',
           'Jalen Hurts', 'Kyler Murray', 'Jonathan Taylor', 'Austin Ekeler',
           'Joe Mixon', 'Najee Harris', 'James Conner', 'Ezekiel Elliott',
           'Nick Chubb', 'Damien Harris', 'Alvin Kamara', 'Antonio Gibson',
           'Leonard Fournette', 'Cordarrelle Patterson', 'Aaron Jones',
           'Derrick Henry', 'Dalvin Cook', 'Josh Jacobs', 'Melvin Gordon',
           'Javonte Williams', 'Devin Singletary', 'David Montgomery',
           'AJ Dillon', 'Darrel Williams', "D'Andre Swift", 'Kirk Cousins',
           'Ryan Tannehill', 'Derek Carr', 'Carson Wentz', 'Lamar Jackson',
           'Russell Wilson', 'Jimmy Garoppolo', 'Cooper Kupp', 'Deebo Samuel',
           'Justin Jefferson', 'Davante Adams', 'Mike Evans',
           'Tyreek Hill', 'Stefon Diggs', 'Mike Williams', 'DK Metcalf',
           'Tyler Lockett', 'Diontae Johnson', 'Hunter Renfrow',
           'Keenan Allen', 'Michael Pittman Jr.', 'Tee Higgins', 'DJ Moore',
           'Chris Godwin', 'Brandin Cooks', 'Jaylen Waddle', 'Darnell Mooney',
           'Amon-Ra St. Brown', 'Terry McLaurin', 'Marquise Brown',
           'Amari Cooper', 'Adam Thielen', 'Christian Kirk', 'Kendrick Bourne',
           'DeVonta Smith', 'Van Jefferson', 'A.J. Brown', "Deshaun Watson",
           "Trey Lance", "Jacoby Brisset", "Kyle Juszczyk",
           "Justin Fields", "Trevor Lawrence", "Tua Tagovailoa",
           "Jameis Winston", "Baker Mayfield", "Jared Goff", "Matt Ryan",
           "Davis Mills", "Mac Jones", "Daniel Jones", "Mitch Trubisky",
           "Kenny Pickett", "Odell Beckham Jr.", "Sterling Shepard",
           "James Connor", "J.K. Dobbins", "Chase Edmonds",
           "Kareem Hunt", "Tony Pollard", "Richard Sherman",
           "Rhamondre Stevenson", "Mark Ingram", "Courtland Sutton",
           "Allen Robinson", "Gabriel Davis", "Rashod Bateman",
           "Allen Lazard", "DeAndre Hopkins",
           "Nico Collins", "Brandon Aiyuk", "Joey Bosa",
           "Kadarius Toney", "Julio Jones",
           "Marquez Valdes-Scantling", "Justin Tucker", "Travis Kelce",
           "Darren Waller", "George Kittle", "Mark Andrews",
           "T.J. Hockenson", "Aaron Donald", "TJ Watt", "JJ Watt",
           "Jamal Adams", "Pat Freiermuth", "Kyle Pitts", "Robert Tonyan",
           "Roquan Smith", "Bobby Wagner", "Devin White", "Derwin James",
           "Jalen Ramsey", "Myles Garret", "Nick Bosa", "Vita Vea"
           "Minkah Fitzpatrick", "Antoine Winfield Jr.", "Xavier McKinney",
           "Tyrann Mathieu", "Budda Baker", "Patrick Peterson", "Jordan Love"]

#Randomly pick players/teams
players_random_sample = random.sample(players,2)
player = players_random_sample[0]
player2 = players_random_sample[1]
team = random.choice(teams)

#create a prompt for the random noun
prompts = [
"write a tweet about a quote from " + player,
"write a quote from " + player + " about how bad " + player2 + " is",
"write a tweet about a hot take about the " + team,
"write a tweet about a hot take about " + player,
"write a tweet about a fun fact about " + player,
"write a tweet about a made up controversial article you just read about the " + team,
"write a tweet about how bad " + player + " is",
"write a tweet about a made up trade involving " + player,
"write a tweet about " + player + " being suspended"
"write a tweet about the " + player + " contract"
]

random_prompt = random.choice(prompts)

#Run the prompt to GPT-3 to generate a tweet
tweet_raw = openai.Completion.create(
    model="text-davinci-003",
    prompt=random_prompt,
    max_tokens=2048,
    top_p=1,
    temperature=0.9,
    frequency_penalty = 2.0
)

#Format the response
tweet = tweet_raw.choices[0].text[2:]

#Tweet out
client.create_tweet(text = tweet)
