#Pip install sleeper-api-wrapper if you don't have it already
from sleeper_wrapper import League
from sleeper_wrapper import Players
from sleeper_wrapper import Stats

import requests
import pandas as pd
import time 
from datetime import datetime
import openai
from timeout_decorator import timeout
from tabulate import tabulate
import warnings

# Filter out all warnings
warnings.filterwarnings('ignore', category=Warning)

import json
# Load the configuration from config.json
with open('config.json') as config_file:
    config_data = json.load(config_file)

### Extract League Info From Sleeper ###

def get_NFL_week():
    #Get the current date
    today = datetime.today()
    #Set the NFL kickoff date
    kickoff = datetime(2023, 9, 7)
    #Calculate the number of days between today and kickoff
    days_since_kickoff = (today - kickoff).days
    #Calculate the number of weeks since kickoff
    weeks_since_kickoff = days_since_kickoff // 7
    
    #Return the current week in the NFL season
    return weeks_since_kickoff + 1

def owners(league):
    #Setup users_df
    users_df = pd.DataFrame(league.get_users())
    owners = config_data['sleeper']['owner_dict']
    #Replace the user_id with the owner name from the owners dictionary
    users_df['owner'] = users_df['user_id'].map(owners)

    #For loop through each row of the dataframe and extract the 'team_name' from the 'metadata' column
    team_names = []
    for index, row in users_df.iterrows():
        # Use get() method to safely access 'team_name' with a default value of 'None'
        team_name = row['metadata'].get('team_name', row['display_name'])
        team_names.append(team_name)

    #Add the 'team_name' column to the dataframe
    users_df['team_name'] = team_names
    owners_df = users_df[['owner', 'display_name','user_id', 'team_name']]
    return owners_df.rename(columns={'user_id': 'owner_id'})

def player_info(players):
    #Get all player information
    players_df = pd.DataFrame(players.get_all_players()).transpose()
    players_df = players_df.loc[(players_df['player_id'] != None) & (players_df['active'] == True)]
    players_df = players_df[['player_id', 'full_name', 'position', 'active', 'team']]
    players_df = players_df.reset_index(drop=True)
    #convert the columns 'player_id' and 'full_name' to a dictionary
    return players_df.set_index('player_id')['full_name'].to_dict()

def determine_result(df):
    result_list = []

    for _, row in df.iterrows():
        # Get the opposing team's total points
        opposing_points = df[(df['matchup_id'] == row['matchup_id']) & (df['owner'] != row['owner'])]['totalPoints'].values[0]

        # Determine the result based on points
        if row['totalPoints'] > opposing_points:
            result_list.append('Win')
        elif row['totalPoints'] < opposing_points:
            result_list.append('Loss')
        else:
            result_list.append('Tie')

    return result_list


def weekly_matchup(week, rosters_df, player_dict, league):
    matchups = league.get_matchups(week = week)
    matchup_df = pd.DataFrame(matchups)
    matchup_df = rosters_df[['roster_id', 'owner']].merge(matchup_df, on = 'roster_id').drop(columns = ['custom_points', 'players', 'players_points'])
    starters = matchup_df.starters.apply(pd.Series)
    starters.columns = ['QB', 'RB1', 'RB2', 'WR1', 'WR2', 'TE', 'FLEX','SF', 'K', 'DEF']
    #Get the starter points for each player
    starter_points = matchup_df.starters_points.apply(pd.Series)
    #Set starter_points column names as starter column names with _points appended
    starter_points.columns = [x + '_PTS' for x in starters.columns]
    matchup_df = pd.concat([matchup_df, starters, starter_points], axis = 1).drop(columns = ['roster_id', 'starters', 'starters_points'],axis = 1)

    #Replace player_id with player name
    columns_to_replace = ['QB', 'RB1', 'RB2', 'WR1', 'WR2', 'TE', 'FLEX', 'SF', 'K']
    for column in columns_to_replace:
        matchup_df[column] = matchup_df[column].map(player_dict)

    matchup_df = matchup_df[['owner', 'matchup_id', 'points', 'QB', 'QB_PTS', 'RB1', 'RB1_PTS', 'RB2', 'RB2_PTS', 'WR1', 'WR1_PTS', 'WR2', 'WR2_PTS', 'TE', 'TE_PTS', 'SF', 'SF_PTS', 'FLEX', 'FLEX_PTS', 'K', 'K_PTS', 'DEF', 'DEF_PTS']]
    matchup_df.sort_values(by = 'matchup_id', inplace = True)
    #Rename the column 'points' to 'totalPoints'
    matchup_df.rename(columns={'points': 'totalPoints'}, inplace=True)
    #Add the 'result' column
    matchup_df['result'] = determine_result(matchup_df)
    # Reorder the columns
    new_column_order = ['matchup_id', 'owner', 'totalPoints', 'result'] + [col for col in matchup_df.columns if col not in ['matchup_id', 'owner', 'totalPoints', 'result']]
    matchup_df = matchup_df[new_column_order]
    return matchup_df.sort_values(by = ['matchup_id', 'totalPoints'], ascending = [True, False])

def rank_playoff_seeds(standings_df):
    #Add division column
    division_dict = config_data['sleeper']['division_dict']
    standings_df['division'] = standings_df['owner'].map(division_dict)
    
    #Find Division Leaders
    division_leaders = standings_df.groupby('division').head(1)
    division_leaders.sort_values(by=['wins', 'points_scored'], ascending=False, inplace=True)
    division_leaders['playoff_seed'] = [1,2,3]
    
    #Remove Division Leaders from standings and find 4th seed
    remaining_teams = standings_df[~standings_df['owner'].isin(division_leaders['owner'])]
    remaining_teams.sort_values(by=['wins', 'points_scored'], ascending=[False, False], inplace=True)
    fourth_seed = remaining_teams.head(1)
    fourth_seed['playoff_seed'] = 4

    #Find the 5th seed
    remaining_teams2 = remaining_teams[~remaining_teams['owner'].isin(fourth_seed['owner'])]
    remaining_teams2.sort_values(by=['points_scored'], ascending=[False], inplace=True)
    fifth_seed = remaining_teams2.head(1)
    fifth_seed['playoff_seed'] = 5

    #Find the 6th seed
    remaining_teams3 = remaining_teams2[~remaining_teams2['owner'].isin(fifth_seed['owner'])]
    remaining_teams3.sort_values(by=['modified_median'], ascending=[False], inplace=True)
    sixth_seed = remaining_teams3.head(1)
    sixth_seed['playoff_seed'] = 6

    #Join the seeds together
    seeds = pd.concat([division_leaders, fourth_seed, fifth_seed, sixth_seed])
    seeds = seeds[['owner', 'playoff_seed']]
    
    return standings_df.merge(seeds, on='owner', how='left').sort_values(by=['playoff_seed', 'wins', 'points_scored'], ascending=[True, False, False])

def highest_scoring_player_sleeper(matchup_df):
    df = matchup_df.copy()
    # List of columns to keep
    columns_to_keep = ['owner']

    # Create an empty list to store data
    transformed_data = []

    # Iterate through each row in the original DataFrame
    for _, row in df.iterrows():
        owner = row['owner']
        matchup_id = row['matchup_id']
        points = row['totalPoints']

        # Iterate through player columns (QB, RB1, RB2, etc.)
        for column in ['QB', 'RB1', 'RB2', 'WR1', 'WR2', 'TE', 'SF', 'FLEX', 'K', 'DEF']:
            player = row[column]
            player_points_col = f"{column}_PTS"
            player_points = row[player_points_col]

            # Append player data to the list
            transformed_data.append([owner, player, player_points])

    # Create a new DataFrame from the transformed data
    transformed_df = pd.DataFrame(transformed_data, columns=['owner', 'player', 'player_points'])

    #Identify the highest scoring player, their owner, and the points scored
    highest_scoring_player = transformed_df.loc[transformed_df['player_points'].idxmax()]
    highest_scoring_player_owner = highest_scoring_player['owner']
    highest_scoring_player_name = highest_scoring_player['player']
    highest_scoring_player_points = highest_scoring_player['player_points']
    return highest_scoring_player_owner, highest_scoring_player_name, highest_scoring_player_points

def highest_scoring_team_sleeper(matchup_df):
    #Find the owner with the highest score
    max_points_index = matchup_df['totalPoints'].idxmax()
    #Use the index to get the owner with the highest points
    owner_with_highest_points = matchup_df.loc[max_points_index, 'owner']
    highest_points = matchup_df.loc[max_points_index, 'totalPoints']
    return owner_with_highest_points, highest_points

def iterate_weeks(week, standings_df, weekly_matchup, rosters_df, player_dict, league):
    #Parse through each week and identify the lowest scoring team and calculate the median points scored
    #Create a new empty dataframe called 'points_scored_df'
    points_scored_df = pd.DataFrame(columns = ['owner', 'week', 'points_scored'])
    for i in range(1, week + 1):
        matchup_df = weekly_matchup(i, rosters_df, player_dict, league)
        #Find the owner with the lowest score
        min_points_index = matchup_df['totalPoints'].idxmin()
        #Use the index to get the owner with the lowest points
        owner_with_lowest_points = matchup_df.loc[min_points_index, 'owner']
        #Add a one to the column 'lowest_scoring_team' for the owner with the lowest points
        standings_df.loc[standings_df['owner'] == owner_with_lowest_points, 'lowest_scoring_team'] += 1

        if i == 1:
            #Add the 'owner' column to the dataframe
            points_scored_df['owner'] = matchup_df['owner']
            #Add the 'week' column to the dataframe
            points_scored_df['week'] = i
            #Add the 'points_scored' column to the dataframe
            points_scored_df['points_scored'] = matchup_df['totalPoints']
        else:
            # Create a new dataframe with the new data
            new_data = pd.DataFrame({
                'owner': matchup_df['owner'],
                'week': i,
                'points_scored': matchup_df['totalPoints']
            })
            # Concatenate the new data with the existing 'points_scored_df'
            points_scored_df = pd.concat([points_scored_df, new_data], ignore_index=True)

        print(' - Week ' + str(i) + ' has been processed')

    # Calculate the modified median points scored (mean of middle 3 scores)
    median_points_scored = points_scored_df.groupby('owner')['points_scored'].apply(
        lambda x: x.nlargest(3).nsmallest(3).mean()).reset_index()
    median_points_scored.rename(columns={'points_scored': 'modified_median'}, inplace=True)

    # Add the 'modified_median_weekly_score' column to the 'standings_df' dataframe
    standings_df = standings_df.merge(median_points_scored, on='owner')
    standings_df['modified_median'] = standings_df['modified_median'].round(0)
    return standings_df.sort_values(by = ['wins', 'points_scored'], ascending = [False, False])

def run_sleeper_weekly(week=None):
    """ 
    This is the main function that will run all of steps to fetch the league and weekly data, compute the scores, bounties and update the standings
    """
    if week is None:
        week = get_NFL_week()
    else:
        week = week
    
    ###Load the League data###
    league = League(config_data['sleeper']['league_id'])
    #Load the rosters
    rosters = league.get_rosters()
    #Load the players
    players = Players()
    #Load the owners
    owners_df = owners(league)
    player_dict = player_info(players)
    #Load the team rosters
    rosters_df = pd.DataFrame(rosters)
    rosters_df = rosters_df[['roster_id', 'owner_id', 'starters','players']]
    rosters_df = owners_df[['owner_id', 'owner']].merge(rosters_df, on='owner_id')
    
    #Set up initial standings
    standings = league.get_standings(league.get_rosters(), league.get_users())
    standings_df = pd.DataFrame(standings)
    #Add column names
    standings_df.columns = ['team_name', 'wins', 'losses', 'points_scored']
    standings_df = owners_df[['owner', 'team_name']].merge(standings_df, on='team_name')
    #Add an empty column called 'lowest_scoring_team'
    standings_df['lowest_scoring_team'] = 0
    print('Loaded the League Data from Sleeper')
    ###Process the weekly matchups, and update the standings###
    #Get the latest weekly matchup
    matchup_df = weekly_matchup(week, rosters_df, player_dict, league)
    updated_standings = iterate_weeks(week, standings_df, weekly_matchup, rosters_df, player_dict, league)
    #Rank the playoff seeds
    ranked_standings = rank_playoff_seeds(updated_standings)

    #Run the Bounty Functions
    HT_Owner, HT_Score = highest_scoring_team_sleeper(matchup_df)
    HP_Owner, HP_Player, HP_Score = highest_scoring_player_sleeper(matchup_df)
    print('Completed processing scores and updating standings')
    fname = 'weekly_scores/week{}_matchup_sleeper.csv'.format(week)
    matchup_df.to_csv(fname, index=False)
    print('Saved week {} matchup data'.format(week))
    return matchup_df, ranked_standings, HT_Owner, HT_Score, HP_Owner, HP_Player, HP_Score

### GPT Summary Generation ###

instruction_prompt = "You are an AI Fantasy Football commissioner tasked with writing a weekly summary to your league mates recapping the latest week of our Dynasty league\n\nI will provide you a table of the weekly matchups, which includes the owners, their matchup_ids (owners with the same matchup IDs are opponents for the week), their players and what they scored, and a standings table with everyone's records. \nRead through scores table and each of the matchups and performances first to understand how each team has done this week and then go through the standings to see how they've been doing for the season. Once you've reviewed all of this information, write an email recapping the performances of teams and players. In particular, make sure to roast of the team with lowest total points). \nMake the tone funny, light-hearted and slightly sarcastic"

def get_completion(instruction_prompt, input_prompt, model = "gpt-4"):
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                "role": "system",
                "content": instruction_prompt
                },
                {
                "role": "user",
                "content": input_prompt
                }
            ],
            temperature=1,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
            )
        
        return response['choices'][0]['message']['content']


def generate_summary(week, matchup_df, updated_standings, model = "gpt-4"):
    openai.api_key = config_data['openai']['api_key']
    
    #Convert the tables to be ingested into the prompt
    matchup_tabulate = tabulate(matchup_df, headers='keys', tablefmt='plain', showindex=False)
    standings_tabulate = tabulate(updated_standings, headers='keys', tablefmt='plain', showindex=False)
    
    #Set up the prompt
    input_prompt = f"Week:{week}, \nMatchup Table: \n\n{matchup_tabulate}\n\nStandings Table: \n\n{standings_tabulate}\n\nSummary:"

    #Generate the summary
    summary = get_completion(instruction_prompt, input_prompt, model = model)
    print('Generated the summary')
    return summary

### Email Sending ###
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load the user name and password from yaml file
user, password = config_data['gmail']['GMAIL_USER'], config_data['gmail']['GMAIL_PW']

def send_email(user, week, summary, standings, HT_Owner, HT_Score, HP_Owner, HP_Player, HP_Score):
    # Initialize the server
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login(user, password)
    
    # Define message
    subject = f"Dynasty FF Week {week} Recap"
    #Convert the summary to html
    summary_html = summary.replace('\n', '<br>')
    body = f"""\
    <html>
    <head></head>
    <body>
        <h1>{subject}</h1>
        <br>{summary_html}</br>
        <br>
        <hr>
        <p>Highest Scoring Team: {HT_Owner}: {HT_Score} points</p>
        <p>Highest Scoring Player: {HP_Owner} - {HP_Player}: {HP_Score} Points</p>
        <br>
        {standings}
    </body>
    </html>
    """

    recipients = list(config_data['emails']['sleeper_email_dict'].values())
    sender = user
    msg = MIMEMultipart("alternative")
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject

    # Attach the HTML body
    msg.attach(MIMEText(body, 'html'))

    # Send email
    s.sendmail(sender, recipients, msg.as_string())
    print('The email has been sent')
    # Terminate the session
    s.quit()

def run_sleeper_weekly_email(week=None):
    """ 
    Complete function to run all of the steps to fetch the league and weekly data, compute the scores, bounties and update the standings, generate the summary, and send the email
    """
    if week is None:
        week = get_NFL_week()
    else:
        week = week

    #Extract the data from Sleeper
    matchup_df, updated_standings, HT_Owner, HT_Score, HP_Owner, HP_Player, HP_Score = run_sleeper_weekly(week)

    #Generate the summary
    summary = generate_summary(week, matchup_df, updated_standings)

    #Send the email
    send_email(user, week, summary, updated_standings.to_html(), HT_Owner, HT_Score, HP_Owner, HP_Player, HP_Score)

# Main execution block
if __name__ == "__main__":
    run_sleeper_weekly_email()