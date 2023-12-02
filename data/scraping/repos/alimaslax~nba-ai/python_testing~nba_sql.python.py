import os
from dotenv import load_dotenv
import openai
import mysql.connector
from flask import Flask, request, jsonify

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

prefix = '''### MySql SQL tables, with their properties:
# Draft(yearDraft, numberPickOverall, numberRound, numberRoundPick, namePlayer, slugTeam, nameOrganizationFrom, typeOrganizationFrom, idPlayer, idTeam, nameTeam, cityTeam, teamName, PLAYER_PROFILE_FLAG, slugOrganizationTypeFrom, locationOrganizationFrom)
# Player(id, full_name, first_name, last_name, is_active)
# Player_Attributes(ID, FIRST_NAME, LAST_NAME, DISPLAY_FIRST_LAST, DISPLAY_LAST_COMMA_FIRST, DISPLAY_FI_LAST, PLAYER_SLUG, BIRTHDATE, SCHOOL, COUNTRY, LAST_AFFILIATION, HEIGHT, WEIGHT, SEASON_EXP, JERSEY, POSITION, ROSTERSTATUS, GAMES_PLAYED_CURRENT_SEASON_FLAG, TEAM_ID, TEAM_NAME, TEAM_ABBREVIATION, TEAM_CODE, TEAM_CITY, PLAYERCODE, FROM_YEAR, TO_YEAR, DLEAGUE_FLAG, NBA_FLAG, GAMES_PLAYED_FLAG, DRAFT_YEAR, DRAFT_ROUND, DRAFT_NUMBER, PTS, AST, REB, ALL_STAR_APPEARANCES, PIE)
# Game(GAME_ID, SEASON_ID, TEAM_ID_HOME, TEAM_ABBREVIATION_HOME, TEAM_NAME_HOME, GAME_DATE, MATCHUP_HOME, WL_HOME, MIN_HOME, FGM_HOME, FGA_HOME, FG_PCT_HOME, FG3M_HOME, FG3A_HOME, FG3_PCT_HOME, FTM_HOME, FTA_HOME, FT_PCT_HOME, OREB_HOME, DREB_HOME, REB_HOME, AST_HOME, STL_HOME, BLK_HOME, TOV_HOME, PF_HOME, PTS_HOME, PLUS_MINUS_HOME, VIDEO_AVAILABLE_HOME, TEAM_ID_AWAY, TEAM_ABBREVIATION_AWAY, TEAM_NAME_AWAY, MATCHUP_AWAY, WL_AWAY, MIN_AWAY, FGM_AWAY, FGA_AWAY, FG_PCT_AWAY, FG3M_AWAY, FG3A_AWAY, FG3_PCT_AWAY, FTM_AWAY, FTA_AWAY, FT_PCT_AWAY, OREB_AWAY, DREB_AWAY, REB_AWAY, AST_AWAY, STL_AWAY, BLK_AWAY, TOV_AWAY, PF_AWAY, PTS_AWAY, PLUS_MINUS_AWAY, VIDEO_AVAILABLE_AWAY, GAME_DATE_EST, GAME_SEQUENCE, GAME_STATUS_ID, GAME_STATUS_, GAMECODE, HOME_TEAM_ID, VISITOR_TEAM_ID, SEASON, LIVE_PERIOD, LIVE_PC_TIME, NATL_TV_BROADCASTER_ABBREVIATION, LIVE_PERIOD_TIME_BCAST, WH_STATUS, TEAM_CITY_HOME, PTS_PAINT_HOME, PTS_2ND_CHANCE_HOME, PTS_FB_HOME, LARGEST_LEAD_HOME, LEAD_CHANGES_HOME, TIMES_TIED_HOME, TEAM_TURNOVERS_HOME, TOTAL_TURNOVERS_HOME, TEAM_REBOUNDS_HOME, PTS_OFF_TO_HOME, TEAM_CITY_AWAY, PTS_PAINT_AWAY, PTS_2ND_CHANCE_AWAY, PTS_FB_AWAY, LARGEST_LEAD_AWAY, LEAD_CHANGES_AWAY, TIMES_TIED_AWAY, TEAM_TURNOVERS_AWAY, TOTAL_TURNOVERS_AWAY, TEAM_REBOUNDS_AWAY, PTS_OFF_TO_AWAY, LEAGUE_ID, GAME_DATE_DAY, ATTENDANCE, GAME_TIME, TEAM_CITY_NAME_HOME, TEAM_NICKNAME_HOME, TEAM_WINS_LOSSES_HOME, PTS_QTR1_HOME, PTS_QTR2_HOME, PTS_QTR3_HOME, PTS_QTR4_HOME, PTS_OT1_HOME, PTS_OT2_HOME, PTS_OT3_HOME, PTS_OT4_HOME, PTS_OT5_HOME, PTS_OT6_HOME, PTS_OT7_HOME, PTS_OT8_HOME, PTS_OT9_HOME, PTS_OT10_HOME, PTS_HOME_y, TEAM_CITY_NAME_AWAY, TEAM_NICKNAME_AWAY, TEAM_WINS_LOSSES_AWAY, PTS_QTR1_AWAY, PTS_QTR2_AWAY, PTS_QTR3_AWAY, PTS_QTR4_AWAY, PTS_OT1_AWAY, PTS_OT2_AWAY, PTS_OT3_AWAY, PTS_OT4_AWAY, PTS_OT5_AWAY, PTS_OT6_AWAY, PTS_OT7_AWAY, PTS_OT8_AWAY, PTS_OT9_AWAY, PTS_OT10_AWAY, LAST_GAME_ID, LAST_GAME_DATE_EST, LAST_GAME_HOME_TEAM_ID, LAST_GAME_HOME_TEAM_CITY, LAST_GAME_HOME_TEAM_NAME, LAST_GAME_HOME_TEAM_ABBREVIATION, LAST_GAME_HOME_TEAM_POINTS, LAST_GAME_VISITOR_TEAM_ID, LAST_GAME_VISITOR_TEAM_CITY, LAST_GAME_VISITOR_TEAM_NAME, LAST_GAME_VISITOR_TEAM_CITY1, LAST_GAME_VISITOR_TEAM_POINTS, HOME_TEAM_WINS, HOME_TEAM_LOSSES, SERIES_LEADER, VIDEO_AVAILABLE_FLAG, PT_AVAILABLE, PT_XYZ_AVAILABLE, HUSTLE_STATUS, HISTORICAL_STATUS)
# Team(id, full_name, abbreviation, nickname, city, state, year_founded)
# Team_Attributes(ID, ABBREVIATION, NICKNAME, YEARFOUNDED, CITY, ARENA, ARENACAPACITY, OWNER, GENERALMANAGER, HEADCOACH, DLEAGUEAFFILIATION, FACEBOOK_WEBSITE_LINK, INSTAGRAM_WEBSITE_LINK, TWITTER_WEBSITE_LINK)
### A sql query to find 
'''

# write a function that takes in a string and calls
# openai Completion API
def openai_completion(prompt):
    response = openai.Completion.create(
        model="code-davinci-002",
        prompt=prefix+prompt+"\n SELECT",
        temperature=0,
        max_tokens=150,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["#", ";"]
    )
    # jsonify the response first choice text
    fix_prompt = jsonify(response['choices'][0]['text'])
    # fix = openai.Completion.create(
    #     model="code-davinci-002",
    #     prompt="# fix this sql query #"+response.choices[0].text,
    #     temperature=0,
    #     max_tokens=150,
    #     top_p=1.0,
    #     frequency_penalty=0.0,
    #     presence_penalty=0.0,
    #     stop=["#", ";"]
    # )
    return response


# Connect to MySQL database named main running on localhost with user root and password lewenberg
db = mysql.connector.connect(host="localhost", user="root", passwd="lewenberg", database="main")

# create whatever objects is needed to run a sql query
cursor = db.cursor()

def execute_sql(sql):
    cursor.execute(sql)
    result = cursor.fetchall()
    return pp(cursor,result)

# a function that pretty prints cursor
def pp(cursor, data=None, rowlens=0):
    d = cursor.description
    if not d:
        return "#### NO RESULTS ###"
    names = []
    lengths = []
    if not data:
        data = cursor.fetchall(  )
    for dd in d:    # iterate over description
        l = dd[1]
        if not l:
            l = 12             # or default arg ...
        l = max(l, len(dd[0])) # Handle long names
        names.append(dd[0])
        lengths.append(l)
    return [names]+data

# flask example that is listening on port 5000 has a query param route
# named nba-ai that calls the execute_sql function with the given query param
# and returns the result as a json object
app = Flask(__name__)
@app.route('/nba-ai')
def hello_world():
    query = request.args.get('query')
    response = openai_completion(query)
    # print the response from openai
    print(response)
    # return the first choice
    query = response['choices'][0]['text'];
    full_query = "SELECT"+query.split("\n")[0]
    return jsonify(execute_sql(full_query))

# run the flask app
if __name__ == '__main__':
    app.run(port=5000)
