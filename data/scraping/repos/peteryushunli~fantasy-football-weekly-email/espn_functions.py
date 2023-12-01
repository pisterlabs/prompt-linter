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

headers  = {
    'Connection': 'keep-alive',
    'Accept': 'application/json, text/plain, */*',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36',
}

custom_headers = {
 'Connection': 'keep-alive',
 'Accept': 'application/json, text/plain, */*',
 'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36',
 'x-fantasy-filter': '{"filterActive":null}',
 'x-fantasy-platform': 'kona-PROD-1dc40132dc2070ef47881dc95b633e62cebc9913',
 'x-fantasy-source': 'kona'
}

position_mapping = {
 1: 'QB',
 2: 'RB',
 3: 'WR',
 4: 'TE',
 5: 'K',
 16: 'D/ST'
}

# we have a two qb league but QB2 is an OP incase you can roster 2 QBs in a week
eligible_positions = {
 0 : 'QB', 
 2 : 'RB', 
 4 : 'WR',
 6 : 'TE', 
 7 : 'OP',
 16: 'D/ST', 
 17: 'K',
 20: 'Bench', 
 23: 'Flex'
}

### Extract League Info From ESPN ###

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

def get_NFL_season():
    #Get the current date
    today = datetime.today()
    #If the month is between January and July, the season is the previous year
    if today.month < 8:
        return today.year - 1
    #If the month is between August and December, the season is the current year
    else:
        return today.year

def fetch_espn_data(url, params, cookies):
    return requests.get(url, params=params, cookies=cookies)

def load_league(league_id, espn_cookies, year):
    #Load the general league data via the ESPN API
    team_url = 'https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{}/segments/0/leagues/{}?view=mDraftDetail&view=mSettings&view=mTeam&view=modular&view=mNav'
    url = team_url.format(year, league_id)
    r = requests.get(url,headers=headers,cookies=espn_cookies)
    return r.json()

def load_records(league_data):
    record_data = []
    for team in league_data['teams']:
        team_name = team['name'] 
        team_list = [team['id'], team_name, team['record']['overall']['wins'], team['record']['overall']['losses'], team['record']['overall']['pointsFor'], team['record']['overall']['pointsAgainst']]
        record_data.append(team_list)

    owner_dict = config_data['espn']['owner_dict']

    record_df = pd.DataFrame(record_data, columns=['teamId', 'team_name', 'wins', 'losses', 'points_for', 'points_against'])
    record_df['owner'] = record_df['teamId'].map(owner_dict)
    return record_df[['teamId','owner', 'team_name','wins', 'losses', 'points_for', 'points_against']].sort_values(by=['wins', 'points_for'], ascending=False)

def load_schedule(year, league_id, espn_cookies, week):
    url = 'https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{}/segments/0/leagues/{}?view=mMatchupScoreLite'.format(year, league_id)
    r = requests.get(url, cookies=espn_cookies)
    data = r.json()

    schedule = pd.DataFrame(data['schedule'])
    schedule = schedule.loc[schedule['matchupPeriodId'] == week].drop(columns = 'matchupPeriodId')
    #rename the 'id' column in the schedule dataframe to 'matchup_id'
    schedule.rename(columns={'id':'matchup_id'}, inplace=True)
    # Create DataFrames from 'away' and 'home' columns
    away_df = schedule[['away', 'matchup_id']].rename(columns={'away':'team'})
    home_df = schedule[['home', 'matchup_id']].rename(columns={'home':'team'})
    transformed_df = pd.concat([away_df, home_df], ignore_index=True).sort_values(by='matchup_id').reset_index(drop=True)
    dropcolumn = 'pointsByScoringPeriod.{}'.format(week)
    weekly_scores = pd.json_normalize(transformed_df['team']).drop(columns=dropcolumn)
    weekly_scores['matchup_id'] = transformed_df['matchup_id']
    return weekly_scores

def load_weekly_stats(year, league_id, espn_cookies, week):
    #Load the owner dict and convert the keys to integers
    config_data['espn']['owner_dict'] = {int(key): value for key, value in config_data['espn']['owner_dict'].items()}
    owner_dict = config_data['espn']['owner_dict']
    url = 'https://fantasy.espn.com/apis/v3/games/ffl/seasons/{}/segments/0/leagues/{}?view=mMatchup&view=mMatchupScore'.format(year, league_id)
    # create an empty list to append data to
    projection_data = []
    # Define a timeout value in seconds
    timeout_value = 5 

    try:
        #Request data from ESPN
        r = requests.get(url,
                        params={'scoringPeriodId': week},
                        cookies=espn_cookies)
        espn_raw_data = r.json()
        time.sleep(1)
    except TimeoutError:
        # Handle the timeout (e.g., restart or report an error)
        print("Request timed out. Restarting")

    # loop over each team in the request
    for team in espn_raw_data['teams']:
        # get the team_id so we can map it to team_names
        team_id = team['id']
        # loop over every player on the teams roster
        for player in team['roster']['entries']:
            player_name = player['playerPoolEntry']['player']['fullName']
            lineup_slot = player['lineupSlotId']
            # create a new column with the position using the lineup_slot as the key
            #position = eligible_positions[lineup_slot]
            
            # get the projected and actual points
            projected = None, 
            actual = None
            # loop over the stats for each player
            for stats in player['playerPoolEntry']['player']['stats']:
                # skip the rows where the scoring period does not match up with the current week
                if stats['scoringPeriodId'] != week:
                    continue
                # if the source id = 0 then these are actual stats
                if stats['statSourceId'] == 0:
                    actual = stats['appliedTotal']
                # if the source id = 1 then these are projected stats
                elif stats['statSourceId'] == 1:
                    projected = stats['appliedTotal']
            
            # append all the data to the empty list
            projection_data.append([
                week, 
                team_id, 
                player_name, 
                lineup_slot, 
                projected, 
                actual
                ])

    # convert the list to a dataframe with the following column names
    weekly_df = pd.DataFrame(projection_data)
    #Set column names
    weekly_df.columns = ['week', 'teamId', 'player', 'lineup_slot', 'projected', 'actual']
    #Map the team id to the owner name
    weekly_df['owner'] = weekly_df['teamId'].map(owner_dict)
    #Map the lineup slot to the position
    weekly_df['lineup_slot'] = weekly_df['lineup_slot'].map(eligible_positions)
    #Filter out the bench players
    weekly_df = weekly_df[weekly_df['lineup_slot'] != 'Bench']
    weekly_df = weekly_df[weekly_df['lineup_slot'].notna()]
    return weekly_df

def modify_positions(df):
    # Function to modify lineup_slot based on positions
    positions_to_modify = ['RB', 'WR']
    for position in positions_to_modify:
        count = 0
        for i, row in df.iterrows():
            if row['lineup_slot'] == position:
                count += 1
                df.at[i, 'lineup_slot'] = f'{position}{count}'
    return df

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

def transform_weekly(weekly_df, schedule_df):
    #Merge with schedule
    weekly_df = schedule_df.merge(weekly_df, on=['teamId'], how='left')

    #Clean the positions
    weekly_df = weekly_df.groupby('owner').apply(modify_positions).reset_index(drop=True)
    
    #Transform the dataframe
    #Create pivot tables for each position
    pivot_tables = {}
    positions = weekly_df['lineup_slot'].unique()
    #remove nan from the list
    positions = positions[~pd.isnull(positions)]

    for position in positions:
        pivot_table = weekly_df[weekly_df['lineup_slot'] == position].pivot_table(
            values=['player', 'actual'],
            index=['teamId', 'owner'],
            columns=['lineup_slot'],
            aggfunc='first'
        )
        pivot_tables[position] = pivot_table

    # Merge pivot tables into one
    result_df = pd.concat(pivot_tables.values(), axis=1)
    # Reset column levels and rename columns
    result_df.columns = result_df.columns.map(lambda x: '_'.join(x))
    result_df.reset_index(inplace=True)

    #For every column that starts with 'player_', remove the 'player_' and for columns that start with 'actual_', remove the 'actual_' and add '_PTS' to the end
    result_df.columns = result_df.columns.map(lambda x: x.replace('player_', '').replace('actual_', '') + '_PTS' if x.startswith('actual_') else x.replace('player_', ''))
    matchup_df = schedule_df.merge(result_df, how='left', on='teamId')
    #rename the 'points' column to 'total_points'
    matchup_df.rename(columns={'points':'totalPoints'}, inplace=True)
    matchup_df['result'] = determine_result(matchup_df)
    #Reorder the columns
    matchup_df = matchup_df[['matchup_id', 'teamId', 'owner', 'totalPoints', 'QB', 'QB_PTS', 'RB1', 'RB1_PTS', 'RB2', 'RB2_PTS', 'WR1', 'WR1_PTS', 'WR2', 'WR2_PTS', 'TE', 'TE_PTS', 'Flex', 'Flex_PTS', 'D/ST', 'D/ST_PTS', 'K', 'K_PTS']]
    return matchup_df


def highest_scoring_player_espn(weekly_df):
    #Find the highest scoring player
    highest_player = weekly_df.loc[weekly_df['actual'].idxmax()]
    return highest_player.owner, highest_player.player, highest_player.actual

def highest_scoring_team_espn(weekly_df):
    #Find the highest scoring team
    highest_team = weekly_df.groupby('owner').sum().sort_values(by='actual', ascending=False).head(1)
    return highest_team.index[0], highest_team['actual'][0]
    
def iterate_weeks_espn(year, week, standings_df, league_id, espn_cookies):
    #Lowest Scoring Team column and set it to 0
    standings_df['lowest_scoring_team'] = 0

    #Create points scored df
    points_scored_df = pd.DataFrame(columns = ['owner', 'week', 'points_scored'])
    
    #Parse through each week and identify the lowest scoring team and record the points scored
    for i in range(1, week + 1):
        weekly_df = load_weekly_stats(year, league_id, espn_cookies,i)
        weekly_df = weekly_df[weekly_df['lineup_slot'] != 'Bench']
        matchup_df = weekly_df.groupby('owner')['actual'].sum().reset_index()
        #Find the owner with the lowest score
        min_points_index = matchup_df['actual'].idxmin()
        #Use the index to get the owner with the lowest points
        owner_with_lowest_points = matchup_df.loc[min_points_index, 'owner']
        
        #Add a one to the column 'lowest_scoring_team' for the owner with the lowest points
        standings_df.loc[standings_df['owner'] == owner_with_lowest_points, 'lowest_scoring_team'] += 1

        #Update the points_scored_df
        append = pd.DataFrame({'owner': matchup_df['owner'], 'week': i, 'points_scored': matchup_df['actual']})
        points_scored_df = pd.concat([points_scored_df, append], ignore_index=True)

    # Calculate the modified median points scored (mean of middle 3 scores)
    median_points_scored = points_scored_df.groupby('owner')['points_scored'].apply(
        lambda x: x.nlargest(3).nsmallest(3).mean()).reset_index()
    median_points_scored.rename(columns={'points_scored': 'modified_median'}, inplace=True)

    # Add the 'modified_median_weekly_score' column to the 'standings_df' dataframe
    standings_df = standings_df.merge(median_points_scored, on='owner').sort_values(by=['wins', 'points_for'],ascending=False)
    standings_df['modified_median'] = standings_df['modified_median'].round(0)
    return standings_df

def rank_playoff_seeds(standings_df):
    # Sort the standings by Wins and then by points_for in descending order
    standings_df = standings_df.sort_values(by=['wins', 'points_for'], ascending=[False, False])
    
    # Assign seeds to the top 5 teams
    top5 = standings_df[['teamId']].head(5)
    
    # Determine the 6th seed based on the remaining team with the highest points_for
    remaining_teams = standings_df.tail(9)
    sixth_seed = remaining_teams.sort_values(by='points_for', ascending=False)[['teamId']].head(1)
    
    # Determine the 7th seed based on the remaining team with the highest median_weekly_score
    remaining_teams = remaining_teams[~remaining_teams['teamId'].isin(sixth_seed['teamId'])]
    seventh_seed = remaining_teams.sort_values(by='modified_median', ascending=False)[['teamId']].head(1)
    
    # Create a DataFrame for the playoff seeds
    playoff_teams = pd.concat([top5, sixth_seed, seventh_seed]).reset_index(drop=True)
    #Create a column for the playoff seeds
    playoff_teams['playoff_seed'] = [1, 2, 3, 4, 5, 6, 7]

    # Merge the playoff seeds DataFrame with the standings DataFrame
    standings_df = standings_df.merge(playoff_teams, on='teamId', how='left')
    
    return standings_df.sort_values(by='playoff_seed').reset_index(drop=True)

def run_espn_weekly(week=None, year=None, max_attempts=5):
    attempts = 0
    if year is None:
                year = get_NFL_season()
    else:
        year = year

    if week is None:
        week = league_data['scoringPeriodId'] - 1
    else:
        week = week

    espn_cookies = {
        "swid": config_data['espn']['swid'],
        "espn_s2": config_data['espn']['espn_s2']
    }
    
    league_id = config_data['espn']['league_id']
    league_data = load_league(league_id, espn_cookies, year)
    owner_dict = config_data['espn']['owner_dict']

    while attempts < max_attempts:
        #Load the league data via the ESPN API
        standings_df = load_records(league_data)
        weekly_df = load_weekly_stats(year, league_id, espn_cookies, week)
        schedule_df = load_schedule(year, league_id, espn_cookies, week)

        matchup_df = transform_weekly(weekly_df, schedule_df)
        standings_df = iterate_weeks_espn(year, week, standings_df, league_id, espn_cookies)
        standings_df = rank_playoff_seeds(standings_df)
        matchup_df = standings_df[['teamId', 'team_name']].merge(matchup_df, on='teamId', how='left').sort_values(by='matchup_id').reset_index(drop=True)

        if (len(matchup_df) > 0) and (len(schedule_df) > 0) and (len(standings_df) > 0):
            print('Loaded data from ESPN for week ' + str(week) + ' of the ' + str(year) + ' season')
            break  # Break the loop if data is loaded correctly
        else:
            attempts += 1
            print(f'Data not loaded correctly (Attempt {attempts} of {max_attempts}), retrying in 1 second...')
            time.sleep(1)  # Wait for 1

    if attempts == max_attempts:
        print(f'Failed to load data after {attempts} attempts. Please check your configuration.')

    HP_Owner, HP_Player, HP_Score = highest_scoring_player_espn(weekly_df)
    HT_Owner, HT_Score = highest_scoring_team_espn(weekly_df)
    print('Completed processing weekly scores and standings')

    fname = fname = 'weekly_scores/week{}_matchup_espn.csv'.format(week)
    #Make adjustments to the matchup_df before saving
    matchup_df['result'] = determine_result(matchup_df)
    matchup_df = matchup_df[['matchup_id', 'owner','team_name', 'totalPoints', 'result', 'QB', 'QB_PTS',
       'RB1', 'RB1_PTS', 'RB2', 'RB2_PTS', 'WR1', 'WR1_PTS', 'WR2', 'WR2_PTS',
       'TE', 'TE_PTS', 'Flex', 'Flex_PTS', 'D/ST', 'D/ST_PTS', 'K', 'K_PTS']]
    matchup_df.to_csv(fname, index=False)
    #print('Saved week {} matchup data'.format(week))
    standings_df.drop(columns=['teamId', 'lowest_scoring_team'], inplace=True)
    return standings_df, matchup_df, HP_Owner, HP_Player, HP_Score, HT_Owner, HT_Score

### GPT Summary Generation ###

instruction_prompt = "You are an AI Fantasy Football commissioner tasked with writing a weekly summary to your league mates recapping the latest week of Fantasy Football.\n\nI will provide you a weekly scores table with each matchup (rows with the same matchup_id played each other and the total_points determines the winner), the owners, their players and what they scored. There will also be a standings table with everyone's records. \nRead through scores table and each of the matchups and performances first to understand how each team has done this week and then go through the standings to see how they've been doing for the season. Once you've reviewed all of this information, write an email recapping the performances of teams and players. In particular, make sure to roast of the team with lowest total points). \nMake the tone funny, light-hearted and slightly sarcastic"

def get_completion(instruction_prompt, input_prompt, model = 'gpt-3.5-turbo'):
        response = openai.ChatCompletion.create(
            model= model,
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

def generate_summary(week, matchup_df, standings_df, model = 'gpt-4'):
    openai.api_key = config_data['openai']['api_key']
    #Convert the input tables for easier ingestion
    scoring_tabulate = tabulate(matchup_df, headers='keys', tablefmt='plain', showindex=False)
    standings_tabulate = tabulate(standings_df, headers='keys', tablefmt='plain', showindex=False)

    #Create the input prompt
    input_prompt = f"Week:{week}, \nScores Table: \n\n{scoring_tabulate}\n\nStandings Table: \n\n{standings_tabulate}\n\nSummary:"
    #Generate the summary
    summary = get_completion(instruction_prompt, input_prompt, model = model)
    print('Generated the summary')
    return summary

### Send the Email ###
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load the user name and password from yaml file
user, password = config_data['gmail']['GMAIL_USER'], config_data['gmail']['GMAIL_PW']

def send_email(user, week, summary, standings, HT_Owner, HT_Score, HP_Owner, HP_Player, HP_Score, league_name = None):
    # Initialize the server
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login(user, password)
    
    if league_name is None:
        subject = f"Fantasy Football Week {week} Recap email"
    else:
        subject = f"{league_name}: Week {week} Recap"
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

    recipients = list(config_data['emails']['espn_email_dict'].values())
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


def run_sleeper_weekly_espn(week=None):
    """ 
    Complete function to run all of the steps to fetch the league and weekly data, compute the scores, bounties and update the standings, generate the summary, and send the email
    """
    if week is None:
        week = get_NFL_week()
    else:
        week = week

    

    #Extract the data from Sleeper
    matchup_df, updated_standings, HT_Owner, HT_Score, HP_Owner, HP_Player, HP_Score = run_sleeper_weekly_espn(week)

    #Generate the summary
    summary = generate_summary(week, matchup_df, updated_standings)

    #Send the email
    send_email(user, week, summary, updated_standings.to_html(), HT_Owner, HT_Score, HP_Owner, HP_Player, HP_Score)

# Main execution block
if __name__ == "__main__":
    run_sleeper_weekly_espn()