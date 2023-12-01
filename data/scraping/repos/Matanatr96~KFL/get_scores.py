import argparse
from collections import defaultdict

import openai
import pandas as pd
from sleeper_wrapper import League
from dotenv import dotenv_values

config = dotenv_values(".env")
openai.api_key = config['OPENAI_API_KEY']
league = League(config['LEAGUE_ID'])

roster_conversions = ['Mattapalli', 'Komaragiri', 'Idate', 'Bada', 'Digby', 'Nethi', 'Rattan', 'Upadhyaya', 'Aireddy',
                      'Hansen', 'Le', 'Pandya']

parser = argparse.ArgumentParser()
parser.add_argument('-w', '--week', required=True)
args = parser.parse_args()


def get_matchups(week: int) -> (defaultdict, pd.DataFrame):
    all_matchups = league.get_matchups(week)
    matchups = defaultdict(list)
    ranks = []
    for matchup in all_matchups:
        matchup_id = matchup['matchup_id']
        matchups[matchup_id].append((matchup['roster_id'], matchup['points']))
        ranks.append([matchup['roster_id'], matchup['points']])

    ranks_df = pd.DataFrame(ranks, columns=['Id', 'Score']).sort_values(by='Score', ascending=False).set_index('Id')
    ranks_df['Rank'] = range(1, 13)
    return matchups, ranks_df


def get_scores(week: int) -> list:
    matchups, ranks = get_matchups(week)
    all_scores = []
    for j in matchups.values():
        roster_id1 = j[0][0]
        roster_id2 = j[1][0]
        one_score = j[0][1]
        two_score = j[1][1]
        one_diff = round(one_score - two_score, 2)
        two_diff = round(two_score - one_score, 2)
        all_scores.extend(
            (
                [
                    2023,
                    roster_conversions[roster_id1 - 1],
                    week,
                    1 if one_score > two_score else 0,
                    one_score,
                    two_score,
                    roster_conversions[roster_id2 - 1],
                    one_diff,
                    ranks.loc[roster_id1, 'Rank'],
                ],
                [
                    2023,
                    roster_conversions[roster_id2 - 1],
                    week,
                    1 if one_score < two_score else 0,
                    two_score,
                    one_score,
                    roster_conversions[roster_id1 - 1],
                    two_diff,
                    ranks.loc[roster_id2, 'Rank'],
                ],
            )
        )
    return all_scores


def chat_gpt_format(scores: list) -> str:
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                              messages=[{"role": "system", "content": f"""Convert the following array 
                                              of arrays into text I can copy into excel with newlines after each line 
                                              and remove the quotations: {scores}"""}])
    return completion.choices[0].message.content


if __name__ == '__main__':
    df = get_scores(args.week)
    formatted_scores = chat_gpt_format(df)
    print(formatted_scores)
