import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import openai

from sklearn.decomposition import PCA
df_players = pd.read_csv("/Users/buinhatquang/Desktop/playercomparison/website/data/players.csv")
df_radar = pd.read_csv("/Users/buinhatquang/Desktop/playercomparison/website/data/radar.csv")

all_players = list(df_players["Player"].values)
openai.api_key = "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"

forward_position = ["FW", "FWMF", "FWDF"]
midfield_position = ["MF", "MFDF", "MFFW"]
defender_position = ["DF", "DFMF", "DFFW"]

forward_category = ["Ball carrier", "Goal scorer", "Well rounded", "Injury prone / Sub"]
midfield_category = ["Defensive-minded midfielder", "Attacking/Ball carrying midfielder", "Dynamic midfielder", "Injury prone / Sub"]
defend_category = ["Well rounded defender", "Aggressive defender", "Ball-playing defender", "Injury prone / Sub"]
color_ranking = ["steelblue", "green", "gold", "red"]
general_info = ["Player", "Nation", "Pos", "Squad", "Age", "Born"]
playing_time = ["MP", "Starts", "Min"]
goals_bestfeatures = ["Goals", "Shots", "SoT"]
passes_bestfeatures = ["PasTotCmp", "Assists"]
skill_bestfeatures = ["SCA", "ScaDrib"]
defense_bestfeatures = ["Tkl", "Int"]
forward_features_reduced = ["Goals", "Shots", "SCA", "Assists", "Car3rd", "ScaFld", "Carries", "CarTotDist", "CarPrgDist", 'CPA', "ScaDrib"]
midfielder_features_reduced = ["Goals","PasTotCmp", "Assists", "PasAss", "Pas3rd", "Crs", "PasCmp", "SCA", "ScaDrib", "GCA", "Tkl", "TklWon", "TklDri", 
                       "TklDriAtt", "Blocks", "BlkSh", "Int", "Recov", "Carries", "CarPrgDist" , "Fld"]
defender_features_reduced = ["PasTotCmp", "PasTotDist", "PasTotPrgDist", "Tkl", "TklWon", "TklDef3rd", "TklMid3rd", "TklAtt3rd", "TklDri", "TklDriAtt", "TklDriPast", "Blocks", 
                     "BlkSh", "Int", "Tkl+Int", "Recov", "AerWon", "AerLost", "Carries", "CarTotDist", "CarPrgDist", "CrdY", "CrdR","Fls", "Clr"]

forward_features = ["Goals", "Shots", "SoT", "G/Sh", "G/SoT", "ShoDist", "GCA", "SCA", "Off", "PKwon", "ScaDrib", "Assists",
                    "ScaPassLive", "Car3rd", "ScaFld", "ToAtt", "ToSuc", "Carries", "CarTotDist", "CarPrgDist", 'CPA', "CarMis", "CarDis"]
midfielder_features = ["Goals","PasTotCmp", "PasTotCmp%", "PasTotDist", "PasTotPrgDist", "Assists", "PasAss", "Pas3rd", "Crs", "PasCmp", 
                       "PasOff", "PasBlocks", "SCA", "ScaPassLive", "ScaPassDead", "ScaDrib", "ScaSh", "ScaFld", "GCA", "GcaPassLive", 
                       "GcaPassDead", "GcaDrib", "GcaSh", "GcaFld", "Tkl", "TklWon", "TklDef3rd", "TklMid3rd", "TklAtt3rd", "TklDri", 
                       "TklDriAtt", "TklDri%", "TklDriPast", "Blocks", "BlkSh", "Int", "Recov", "Carries", "CarTotDist", "CarPrgDist" , "Fld"]
defender_features = ["PasTotCmp", "PasTotDist", "PasTotPrgDist", "Tkl", "TklWon", "TklDef3rd", "TklMid3rd", "TklAtt3rd", "TklDri", "TklDriAtt", "TklDriPast", "Blocks", 
                     "BlkSh", "Int", "Tkl+Int", "Recov", "AerWon", "AerLost", "Carries", "CarTotDist", "CarPrgDist", "CrdY", "CrdR","Fls", "Clr"]

features_sum = ["Goals", "MP", "Starts", "Min", "Assists"]


acronyms = {"Rk": "Rank", "Player": "Player's name", "Nation": "Player's nation", "Pos": "Position",
            "Squad": "Squad’s name", "Comp": "League that squat occupies", "Age": "Player's age",
            "Born": "Year of birth", "MP": "Matches played", "Starts": "Matches started",
            "Min": "Minutes played", "90s": "Minutes played divided by 90", 
            "Goals": "Goals scored or allowed", 
            "Shots": "Shots total (Does not include penalty kicks)", 
            "SoT": "Shots on target (Does not include penalty kicks)", 
            'SoT%': 'Shots on target percentage (Does not include penalty kicks)',
            'G/Sh': 'Goals per shot', 
            'G/SoT': 'Goals per shot on target (Does not include penalty kicks)', 
            'ShoDist': 'Average distance, in yards, from goal of all shots taken (Does not include penalty kicks)',
            'ShoFK': 'Shots from free kicks', 
            'ShoPK': 'Penalty kicks made',
            "PasProg":"Completed passes that move the ball towards the opponent's goal at least 10 yards from its furthest point in the last six passes, or any completed pass into the penalty area",
                  "PasAtt":"Passes attempted",
                  "PasLive":"Live-ball passes",
                  "PasDead":"Dead-ball passes",
                  "PasFK":"Passes attempted from free kicks",
                  "TB":"Completed pass sent between back defenders into open space",
                  "Sw":"Passes that travel more than 40 yards of the width of the pitch",
                  "PasCrs":"Crosses",
                  "TI":"Throw-Ins taken",
                  "CK":"Corner kicks",
                  "CkIn":"Inswinging corner kicks",
                  "CkOut":"Outswinging corner kicks",
                  "CkStr":"Straight corner kicks",
                  "PasCmp":"Passes completed",
                  "PasOff":"Offsides",
                  "PasBlocks":"Blocked by the opponent who was standing it the path",
                  "SCA":"Shot-creating actions",
                  "ScaPassLive":"Completed live-ball passes that lead to a shot attempt",
                  "ScaPassDead":"Completed dead-ball passes that lead to a shot attempt",
                  "ScaDrib":"Successful dribbles that lead to a shot attempt",
                  "ScaSh":"Shots that lead to another shot attempt",
                  "ScaFld":"Fouls drawn that lead to a shot attempt",
                  "ScaDef":"Defensive actions that lead to a shot attempt",
                  "GCA":"Goal-creating actions",
                  "GcaPassLive":"Completed live-ball passes that lead to a goal",
                  "GcaPassDead":"Completed dead-ball passes that lead to a goal",
                  "GcaDrib":"Successful dribbles that lead to a goal",
                  "GcaSh":"Shots that lead to another goal-scoring shot",
                  "GcaFld":"Fouls drawn that lead to a goal",
                  "GcaDef":"Defensive actions that lead to a goal",
                  "Tkl":"Number of players tackled",
                  "TklWon":"Tackles in which the tackler's team won possession of the ball",
                  "TklDef3rd":"Tackles in defensive 1/3",
                  "TklMid3rd":"Tackles in middle 1/3",
                  "TklAtt3rd":"Tackles in attacking 1/3",
                  "TklDri":"Number of dribblers tackled",
                  "TklDriAtt":"Number of times dribbled past plus number of tackles",
                  "TklDri%":"Percentage of dribblers tackled",
                  "TklDriPast":"Number of times dribbled past by an opposing player",
                  "Blocks":'Number of times blocking the ball by standing in its path',
                 'BlkSh':'Number of times blocking a shot by standing in its path',
            'PKatt': 'Penalty kicks attempted', 
            'PasTotCmp': 'Passes completed', 
            'PasTotAtt': 'Passes attempted', 
            'PasTotCmp%': 'Pass completion percentage', 
            'PasTotDist': 'Total distance, in yards, that completed passes have traveled in any direction',
            'PasTotPrgDist': 'Total distance, in yards, that completed passes have traveled towards the opponent\'s goal',
            'PasShoCmp': 'Passes completed (Passes between 5 and 15 yards)', 
            'PasShoAtt': 'Passes attempted (Passes between 5 and 15 yards)', 
            'PasShoCmp%': 'Pass completion percentage (Passes between 5 and 15 yards)',
            'PasMedCmp': 'Passes completed (Passes between 15 and 30 yards)', 
            'PasMedAtt': 'Passes attempted (Passes between 15 and 30 yards)', 
            'PasMedCmp%': 'Pass completion percentage (Passes between 15 and 30 yards)',
            'PasLonCmp': 'Passes completed (Passes longer than 30 yards)', 
            'PasLonAtt': 'Passes attempted (Passes longer than 30 yards)', 
            'PasLonCmp%': 'Pass completion percentage (Passes longer than 30 yards)',
            'Assists': 'Assists', 
            'PasAss': 'Passes that directly lead to a shot (assisted shots)', 
            'Pas3rd': 'Completed passes that enter the 1/3 of the pitch closest to the goal',
             "PPA" : "Completed passes into the 18-yard box",
            "CrsPA" : "Completed crosses into the 18-yard box",
           "BlkPass":"Number of times blocking a pass by standing in its path",
                  "Int":"Interceptions",
                  "Tkl+Int":"Number of players tackled plus number of interceptions",
                  "Clr":"Clearances",
                  "Err":"Mistakes leading to an opponent's shot",
                  "Touches":"Number of times a player touched the ball. Note: Receiving a pass, then dribbling, then sending a pass counts as one touch",
                  "TouDefPen":"Touches in defensive penalty area",
                  "TouDef3rd":"Touches in defensive 1/3",
                  "TouMid3rd":"Touches in middle 1/3",
                  "TouAtt3rd":"Touches in attacking 1/3",
                  "TouAttPen":"Touches in attacking penalty area",
                  "TouLive":"Live-ball touches. Does not include corner kicks, free kicks, throw-ins, kick-offs, goal kicks or penalty kicks.",
                  "ToAtt":"Number of attempts to take on defenders while dribbling",
                  "ToSuc":"Number of defenders taken on successfully, by dribbling past them",
                  "ToSuc%":"Percentage of take-ons Completed Successfully",
                  "ToTkl":"Number of times tackled by a defender during a take-on attempt",
                  "ToTkl%":"Percentage of time tackled by a defender during a take-on attempt",
                  "Carries":"Number of times the player controlled the ball with their feet",
                  "CarTotDist":"Total distance, in yards, a player moved the ball while controlling it with their feet, in any direction",
                  "CarPrgDist":"Total distance, in yards, a player moved the ball while controlling it with their feet towards the opponent's goal",
                  "CarProg":"Carries that move the ball towards the opponent's goal at least 5 yards, or any carry into the penalty area",
                  "Car3rd":"Carries that enter the 1/3 of the pitch closest to the goal",
                  "CPA":"Carries into the 18-yard box",
                  "CarMis":"Number of times a player failed when attempting to gain control of a ball",
                  "CarDis":"Number of times a player loses control of the ball after being tackled by an opposing player",
                  "Rec":"Number of times a player successfully received a pass",
           "RecProg" : 
            "Completed passes that move the ball towards the opponents goal at least 10 yards from its furthest point in the last six passes, or any completed pass into the penalty area Make this a dictionary",
           "CrdY":"Yellow cards",
                  "CrdR":"Red cards",
                  "2CrdY":"Second yellow card",
                  "Fls":"Fouls committed",
                  "Fld":"Fouls drawn",
                  "Off":"Offsides",
                  "Crs":"Crosses",
                  "TklW":"Tackles in which the tackler's team won possession of the ball",
                  "PKwon":"Penalty kicks won",
                  "PKcon":"Penalty kicks conceded",
                  "OG":"Own goals",
                  "Recov":"Number of loose balls recovered",
                  "AerWon":"Aerials won",
                  "AerLost":"Aerials lost",
                  "AerWon%":"Percentage of aerials won",
           }




def get_info(player_name, attribute, df):
    '''
    Get information attribute given the player name and a list of attributes
    '''
    if player_name not in all_players:
        return "No player found"
    
    color = list(df[df["Player"] == player_name]["Class"].values)[0]
    return df[df["Player"] == player_name][attribute], color


def attribute_description(attribute):
    '''
    Get the description of each attribute inside the list
    '''
    description = []
    for i in range(len(attribute)):
        description.append(acronyms[attribute[i]])
    return description

def plot_players_right(player_name, attribute, df):
    '''
    Plot the player attribute given the player name, attribute, and desired year
    '''
    
    if player_name not in all_players:
        return "No player found"

    player, color = get_info(player_name, attribute, df)
    description = attribute_description(attribute)
    
    fig = go.Figure(go.Bar(
            x=player.values[0],
            y=description,
            orientation='h',
            marker=dict(
            color=color_ranking[color],
            line=dict(color='black', width=0.5)
        )))
    
    fig.update_layout(
        yaxis_title="Features",
        xaxis=dict(side='top'),
        plot_bgcolor='rgba(0,0,0,0)',
        bargap=0.2,
        height=600
    )
    
    return fig
    
def plot_players_left(player_name, attribute, df):
    '''
    Plot the player attribute given the player name, attribute, and desired year
    '''
    
    if player_name not in all_players:
        return "No player found"
    
    player, color = get_info(player_name, attribute, df)
    description = attribute_description(attribute)
    
    fig = go.Figure(go.Bar(
            x=player.values[0],
            y=description,
            orientation='h',
            marker=dict(
            color=color_ranking[color],
            line=dict(color='black', width=0.5)
        )))

    fig.update_layout(
        xaxis = dict(side='top', range=[max(player.values[0]), 0]),
        yaxis = dict(side='right'),
        plot_bgcolor='rgba(0,0,0,0)',
        bargap=0.2,)

    return fig

def plot_radar(player_name,df):
    '''
    Plot radar chart
    '''
    
    if player_name not in all_players:
        return "No player found"
    
    player_info, color = get_info(player_name, goals_bestfeatures + passes_bestfeatures + skill_bestfeatures + defense_bestfeatures, df)
    
    new_column = attribute_description(player_info.columns)
    player_info.columns = new_column
    
    fig = px.line_polar(player_info, r=list(player_info.values[0]), theta=list(player_info.columns), line_close=True)
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1], showticklabels=False)), plot_bgcolor='white')
    fig.update_traces(fill='toself', fillcolor=color_ranking[color], line_color='black', opacity=0.8)
    return fig

def similar_players(player_name, attribute, df):
    player_info, color = get_info(player_name, attribute + ["Player", "Position Category"], df)
    df_train = df[df["Position Category"]==get_info(player_name, ["Position Category"], df_radar)[0]["Position Category"].values[0]]
    
    cosine_matrix = pd.DataFrame(cosine_similarity(df_train[attribute]))
    cosine_matrix.reset_index(inplace=True, drop=True)
    df_train.reset_index(inplace=True, drop=True)
    df_full = pd.concat([df_train["Player"], cosine_matrix], axis=1)
    
    player_name_list = list(df_full["Player"].values)
    player_coef = list(df_full[df_full["Player"]==player_name].values[0])[1:]
    player_sorted = sorted(zip(player_name_list, player_coef), key=lambda x: x[1], reverse=True)
    players = [x[0] for x in player_sorted]
    
    return players

def player_to_text(player1, player2, attribute):
    player1_text = ""
    player2_text = ""
    for i in range(len(list(df_players[general_info + attribute].columns))):
        player1_text += str(list(df_players[df_players["Player"] == player1][general_info + attribute].columns)[i]) + ": "
        player1_text += str(df_players[df_players["Player"] == player1][general_info + attribute].values[0][i]) + ", "
        player1_text += str(list(df_players[df_players["Player"] == player2][general_info + attribute].columns)[i]) + ": "
        player1_text += str(df_players[df_players["Player"] == player2][general_info + attribute].values[0][i]) + ", "
    return player1_text, player2_text

def compare_stats_between_examples(player1, player2):
    prompt = f"Write an analysis about some of the main given attributes between the following two football players:\n1. {player1}\n2. {player2}\n"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=1500
    )

    return response