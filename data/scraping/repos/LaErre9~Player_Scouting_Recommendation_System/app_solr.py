##### Streamlit Python Application #####

# used libraries 
import os
import json
import pandas as pd
import plotly.graph_objects as go
import re
from typing import List

# solr library
from SolrClient import SolrClient

# bing library for automation image
from bing_image_urls import bing_image_urls

# streamlit libraries
import streamlit as st 
from streamlit_searchbox import st_searchbox

# cosine similarity libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# langchain libraries
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

### Apy Key - GPT 
os.environ['OPENAI_API_KEY'] = "INSERT YOUR KEY"


############### Header #################
# Set the page width to 'wide' to occupy the full width
st.set_page_config(page_title="Player Scouting Recommendation System", page_icon="⚽", layout="wide")
st.markdown("<h1 style='text-align: center;'>⚽🔍 Player Scouting Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>AI-Powered Player Scouting: Scout, Recommend, Elevate Your Team's Game - 👀 This is a Beta version</p>", unsafe_allow_html=True)

############### Simple Search Engine with Auto-Complete Query Suggestion ##############
solr = SolrClient('http://localhost:8983/solr')

press = False
choice = None

# Initialises the streamlit session state useful for page reloading
if 'expanded' not in st.session_state:
    st.session_state.expanded = True

if 'choice' not in st.session_state:
    st.session_state.choice = None

#### Script for Autocomplete
def search_solr(searchterm: str) -> List[any]:
    # Check if a search term is provided
    if searchterm:
        # Query Solr for player names containing the search term
        res = solr.query('FootballStatsCore', {
            'q': 'Player:' + '*' + searchterm + '*',
            'fl': 'Rk,Player',
            'rows': 100000,
        })
        result = res.docs

        # If results are found
        if result != []:
            # Create a DataFrame from the results
            df_p = pd.DataFrame(result)
            # Extract the 'Rk' and 'Player' columns and clean the data
            df_p['Rk'] = df_p['Rk'].apply(lambda x: x[0])
            df_p['Player'] = df_p['Player'].apply(lambda x: x[0])
            # Return the 'Player' column as autocomplete suggestions
            return df_p['Player']
        else:
            # Return an empty list if no results are found
            return []

# Streamlit search box
selected_value = st_searchbox(
    search_solr,
    key="solr_searchbox",
    placeholder="🔍 Search a Football Player"
)

st.session_state.choice = selected_value
choice = st.session_state.choice

################### Organic result ###########################
if choice:
    # Query Solr to retrieve all players data
    res = solr.query('FootballStatsCore', {
        'q': '*:*',
        'fl': '*',
        'rows': 100000,  # Set this value to any large number that exceeds the total number of documents
    })

    result = res.docs

    # Create a DataFrame from the Solr query results
    df_player = pd.DataFrame(result)
    
    # Extract column names from the JSON result
    columns_to_process = list(df_player.columns)

    # Process columns in the DataFrame
    for column in columns_to_process:
        if isinstance(df_player[column][0], (list, dict)):
            df_player[column] = df_player[column].apply(lambda x: x[0] if isinstance(x, list) else (x if isinstance(x, dict) else x))

    # Define columns to drop from the DataFrame
    columns_to_drop = ['id', '_version_']
    df_player = df_player.drop(columns=columns_to_drop)

    # Create a normalized copy of the player DataFrame
    df_player_norm = df_player.copy()

    # Define a custom mapping for the 'Pos' column
    custom_mapping = {
        'GK': 1,
        'DF,FW': 4,
        'MF,FW': 8,
        'DF': 2,
        'DF,MF': 3,
        'MF,DF': 5,
        'MF': 6,
        'FW,DF': 7,
        'FW,MF': 9,
        'FW': 10
    }

    # Apply the custom mapping to the 'Pos' column
    df_player_norm['Pos'] = df_player_norm['Pos'].map(custom_mapping)

    # Select a subset of features for analysis
    selected_features = ['Pos', 'Age', 'Int',
       'Clr', 'KP', 'PPA', 'CrsPA', 'PrgP', 'Playing_Time_MP',
       'Performance_Gls', 'Performance_Ast', 'Performance_G_A',
       'Performance_G-PK', 'Performance_Fls', 'Performance_Fld',
       'Performance_Crs', 'Performance_Recov', 'Expected_xG', 'Expected_npxG', 'Expected_xAG',
       'Expected_xA', 'Expected_A-xAG', 'Expected_G-xG', 'Expected_np_G-xG',
       'Progression_PrgC', 'Progression_PrgP', 'Progression_PrgR',
       'Tackles_Tkl', 'Tackles_TklW', 'Tackles_Def_3rd', 'Tackles_Mid_3rd',
       'Tackles_Att_3rd', 'Challenges_Att', 'Challenges_Tkl_',
       'Challenges_Lost', 'Blocks_Blocks', 'Blocks_Sh', 'Blocks_Pass',
       'Standard_Sh', 'Standard_SoT', 'Standard_SoT_', 'Standard_Sh_90', 'Standard_Dist', 'Standard_FK',
       'Performance_GA', 'Performance_SoTA', 'Performance_Saves',
       'Performance_Save_', 'Performance_CS', 'Performance_CS_',
       'Penalty_Kicks_PKatt', 'Penalty_Kicks_Save_', 'SCA_SCA',
       'GCA_GCA', 
       'Aerial_Duels_Won', 'Aerial_Duels_Lost', 'Aerial_Duels_Won_',
       'Total_Cmp', 'Total_Att', 'Total_Cmp_', 'Total_TotDist',
       'Total_PrgDist', '1_3'
    ]

    ####################### Cosine Similarity #######################################

    # Normalization using Min-Max scaling
    scaler = MinMaxScaler()
    df_player_norm[selected_features] = scaler.fit_transform(df_player_norm[selected_features])

    # Calculate cosine similarity between players based on selected features
    similarity = cosine_similarity(df_player_norm[selected_features])

    # Find the Rk associated with the selected player's name
    index_player = df_player.loc[df_player['Player'] == choice, 'Rk'].values[0]

    # Calculate similarity scores and sort them in descending order
    similarity_score = list(enumerate(similarity[index_player]))
    similar_players = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    # Create a list to store data of similar players
    similar_players_data = []

    # Loop to extract information from similar players
    for player in similar_players[1:11]:  # Exclude the first player (self)
        index = player[0]
        player_records = df_player[df_player['Rk'] == index]
        if not player_records.empty:
            player_data = player_records.iloc[0]  # Get the first row (there should be only one)
            similar_players_data.append(player_data)

    # Create a DataFrame from the data of similar players
    similar_players_df = pd.DataFrame(similar_players_data)

    ########################## Analytics of the player chosen ##########################
    url_player = bing_image_urls(choice+ " "+df_player.loc[df_player['Player'] == choice, 'Squad'].iloc[0]+" 2023", limit=1, )[0]

    with st.expander("Features of The Player selected", expanded=True):

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader(choice)
            st.image(url_player, width=356)

        with col2:
            st.caption("📄 Information of Player")
            col_1, col_2, col_3 = st.columns(3)

            with col_1:
                st.metric("Nation", df_player.loc[df_player['Player'] == choice, 'Nation'].iloc[0], None)
                st.metric("Position", df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0], None)

            with col_2:
                st.metric("Born",df_player.loc[df_player['Player'] == choice, 'Born'].iloc[0],None)
                st.metric("Match Played",df_player.loc[df_player['Player'] == choice, 'Playing_Time_MP'].iloc[0],None, help="In 2022/2023")

            with col_3:
                st.metric("Age",df_player.loc[df_player['Player'] == choice, 'Age'].iloc[0],None)

            st.metric(f"🏆 League: {df_player.loc[df_player['Player'] == choice, 'Comp'].iloc[0]}",df_player.loc[df_player['Player'] == choice, 'Squad'].iloc[0],None, help="In 2022/2023")

        with col3:
            st.caption("⚽ Information target of Player")
            #GK
            if df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "GK":
                col_1, col_2 = st.columns(2)

                with col_1:
                    st.metric("Saves", df_player.loc[df_player['Player'] == choice, 'Performance_Saves'].iloc[0], None)
                    st.metric("Clean Sheet", df_player.loc[df_player['Player'] == choice, 'Performance_CS'].iloc[0], None)

                with col_2:
                    st.metric("Goals Against",df_player.loc[df_player['Player'] == choice, 'Performance_GA'].iloc[0],None)
                    st.metric("ShoTA",df_player.loc[df_player['Player'] == choice, 'Performance_SoTA'].iloc[0],None)
            
            # DF
            if df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "DF" or df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "DF,MF" or df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "DF,FW":
                col_1, col_2, col_3 = st.columns(3)

                with col_1:
                    st.metric("Assist", df_player.loc[df_player['Player'] == choice, 'Performance_Ast'].iloc[0], None)
                    st.metric("Goals", df_player.loc[df_player['Player'] == choice, 'Performance_Gls'].iloc[0], None)

                with col_2:
                    st.metric("Aerial Duel",df_player.loc[df_player['Player'] == choice, 'Aerial_Duels_Won'].iloc[0],None)
                    st.metric("Tackle",df_player.loc[df_player['Player'] == choice, 'Tackles_TklW'].iloc[0],None, help="In 2022/2023")

                with col_3:
                    st.metric("Interception",df_player.loc[df_player['Player'] == choice, 'Int'].iloc[0],None)
                    st.metric("Key Passage",df_player.loc[df_player['Player'] == choice, 'KP'].iloc[0],None)

            # MF
            if df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "MF" or df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "MF,DF" or df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "MF,FW":
                col_1, col_2, col_3 = st.columns(3)

                with col_1:
                    st.metric("Assist", df_player.loc[df_player['Player'] == choice, 'Performance_Ast'].iloc[0], None)
                    st.metric("Goals", df_player.loc[df_player['Player'] == choice, 'Performance_Gls'].iloc[0], None)
                    st.metric("Aerial Duel",df_player.loc[df_player['Player'] == choice, 'Aerial_Duels_Won'].iloc[0],None)

                with col_2:
                    st.metric("GCA",df_player.loc[df_player['Player'] == choice, 'GCA_GCA'].iloc[0],None)
                    st.metric("Progressive PrgP",df_player.loc[df_player['Player'] == choice, 'Progression_PrgP'].iloc[0],None, help="In 2022/2023")

                with col_3:
                    st.metric("SCA",df_player.loc[df_player['Player'] == choice, 'SCA_SCA'].iloc[0],None)
                    st.metric("Key Passage",df_player.loc[df_player['Player'] == choice, 'KP'].iloc[0],None)
            
            # FW
            if df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "FW" or df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "FW,MF" or df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "FW,DF":
                col_1, col_2, col_3 = st.columns(3) 

                with col_1:
                    st.metric("Assist", df_player.loc[df_player['Player'] == choice, 'Performance_Ast'].iloc[0], None)
                    st.metric("Goals", df_player.loc[df_player['Player'] == choice, 'Performance_Gls'].iloc[0], None)
                    st.metric("Aerial Duel",df_player.loc[df_player['Player'] == choice, 'Aerial_Duels_Won'].iloc[0],None)

                with col_2:
                    st.metric("SCA",df_player.loc[df_player['Player'] == choice, 'SCA_SCA'].iloc[0],None)
                    st.metric("xG",df_player.loc[df_player['Player'] == choice, 'Expected_xG'].iloc[0],None)
                    st.metric("xAG",df_player.loc[df_player['Player'] == choice, 'Expected_xAG'].iloc[0],None, help="In 2022/2023")
                    

                with col_3:
                    st.metric("GCA",df_player.loc[df_player['Player'] == choice, 'GCA_GCA'].iloc[0],None)
                    st.metric("Key Passage",df_player.loc[df_player['Player'] == choice, 'KP'].iloc[0],None)
            
                    
    ################# Radar and Rank ######################### 
    col1, col2 = st.columns([1.2, 2])

    with col1:
        ###### Similar Players Component ###############
        st.subheader(f'Top 10 Similar Players to {choice}')
        selected_columns = ["Player", "Nation", "Squad", "Pos", "Age"]
        st.dataframe(similar_players_df[selected_columns], hide_index=True, use_container_width=True)

    with col2:
        ###### Radar Analytics #########################
        categories = ['Performance_Gls', 'Performance_Ast', 'KP', 'GCA_GCA','Aerial_Duels_Won', 'Int', 'Tackles_TklW', 'Performance_Saves', 'Performance_CS', 'Performance_GA','Performance_SoTA']
        selected_players = similar_players_df.head(10)

        fig = go.Figure()

        for index, player_row in selected_players.iterrows():
            player_name = player_row['Player']
            values = [player_row[col] for col in categories]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=player_name
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                )
            ),
            showlegend=True,  
            legend=dict(
                orientation="v", 
                yanchor="top",  
                y=1,  
                xanchor="left",  
                x=1.02,  
            ),
            width=750,  
            height=520  
        )

        st.plotly_chart(fig, use_container_width=True)

    ####################### Scouter AI Component ##################################
    st.header('⚽🕵️‍♂️ Scouter AI')
    message = f"Select the ideal characteristics for your team. Scouter AI will evaluate the most suitable player from the players most similar to {choice}"
    st.caption(message)

    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        with st.form("my_form"):
            st.write("Features of your team")
            # List of game styles and their descriptions
            game_styles = {
                "Tiki-Taka": "This style of play, focuses on ball possession, control, and accurate passing.",
                "Counter-Attack": "Teams adopting a counter-attacking style focus on solid defense and rapid advancement in attack when they regain possession of the ball.",
                "High Press": "This style involves intense pressure on the opposing team from their half of the field. Teams practicing high pressing aim to quickly regain possession in the opponent's area, forcing mistakes under pressure.",
                "Direct Play": "This style of play is more direct and relies on long and vertical passes, often targeting forwards or exploiting aerial play.",
                "Pragmatic Possession": "Some teams aim to maintain ball possession as part of a defensive strategy, slowing down the game pace and limiting opponent opportunities.",
                "Reactive": "In this style, a team adapts to the ongoing game situations, changing their tactics based on what is happening on the field. It can be used to exploit opponent weaknesses or respond to unexpected situations.",
                "Physical and Defensive": "Some teams place greater emphasis on solid defense and physical play, aiming to frustrate opponents and limit their attacking opportunities.",
                "Positional Play": "This style aims to dominate the midfield and create passing triangles to overcome opponent pressure. It is based on player positioning and the ability to maintain ball possession for strategic attacking.",
                "Catenaccio": "This style, originating in Italy, focuses on defensive solidity and counterattacks. Catenaccio teams seek to minimize opponent scoring opportunities, often through zone defense and fast transition play.",
                "Counter Attacking": "This style relies on solid defensive organization and quick transition to attack when the team regains possession of the ball. Forwards seek to exploit spaces left open by the opposing team during the defense-to-attack transition.",
                "Long Ball": "This style involves frequent use of long and direct passes to bypass the opponent's defense. It relies on the physical strength of attackers and can be effective in aerial play situations."
            }

            # List of player experience levels
            player_experience = {
                "Veteran": "A player with a long career and extensive experience in professional football. Often recognized for their wisdom and leadership on the field.",
                "Experienced": "A player with experience, but not necessarily in the late stages of their career. They have solid skills and tactical knowledge acquired over time.",
                "Young": "A player in the early or mid-career, often under 25 years old, with considerable development potential and a growing presence in professional football.",
                "Promising": "A young talent with high potential but still needs to fully demonstrate their skills at the professional level."
            }

            # List of the leagues
            leagues = {
                "Serie A": "Tactical and defensive football with an emphasis on defensive solidity and tactical play.",
                "Ligue 1": "Open games with a high number of goals and a focus on discovering young talents.",
                "Premier League": "Fast-paced, physical, and high-intensity play with a wide diversity of playing styles.",
                "Bundesliga": "High-pressing approach and the development of young talents.",
                "La Liga": "Possession of the ball and technical play with an emphasis on constructing actions."
            }

            # List of formations
            formations = ["4-3-1-2", "4-3-3", "3-5-2", "4-4-2", "3-4-3", "5-3-2", "4-2-3-1","4-3-2-1","3-4-1-2","3-4-2-1"]

            # List of player skills
            player_skills = [
                "Key Passing", "Dribbling", "Speed", "Shooting", "Defending",
                "Aerial Ability", "Tackling", "Vision", "Long Passing", "Agility", "Strength",
                "Ball Control", "Positioning", "Finishing", "Crossing", "Marking",
                "Work Rate", "Stamina", "Free Kicks", "Leadership","Penalty Saves","Reactiveness","Shot Stopping",
                "Off the Ball Movement", "Teamwork", "Creativity", "Game Intelligence"
            ]

            ######### Inside FORM #####################
            st.subheader("Select a game style:")
            selected_game_style = st.selectbox("Choose a game style:", list(game_styles.keys()))

            st.subheader("Select player type:")
            selected_player_experience = st.selectbox("Choose player type:", list(player_experience.keys()))

            st.subheader("Select league:")
            selected_league = st.selectbox("Choose a league:", list(leagues.keys()))

            st.subheader("Select formation:")
            selected_formation = st.selectbox("Choose a formation:", formations)

            st.subheader("Select player skills:")
            selected_player_skills = st.multiselect("Choose player skills:", player_skills)

            form = st.form_submit_button("➡️ Confirm features")

    with col2:

        ######### Inside REPORT #####################
        st.info('The text is generated by a GPT-3.5 artificial intelligence model. Please note that the accuracy and veracity of the content may vary. \
            The primary goal is to provide general information and assistance in choosing a football player, but it is always recommended to verify and confirm any information from reliable sources.', icon="ℹ️")

        if form:
            st.caption("Selected Options:")
            st.write(f"You have chosen a game style: {selected_game_style}. {game_styles[selected_game_style]} \
            This player must be {selected_player_experience} and have a good familiarity with the {selected_formation} and the skills of: {', '.join(selected_player_skills)}.")

            template = (
                """You are a soccer scout and you must be good at finding the best talents in your team starting from the players rated by the similar player system."""
            )
            system_message_prompt = SystemMessagePromptTemplate.from_template(template)

            human_template = """
                Generate a Football Talent Scout report based on the DATA PROVIDED (maximum 250 words) written in a formal tone FOLLOWING THE EXAMPLE.
                It is essential to compare player attributes and select the most suitable candidate from the available options from among similar players, based on the TEAM REQUIREMENTS provided. It is important to note that the selection of players is not limited to the ranking of the players provided, as long as they meet the TEAM REQUIREMENTS.
                THE PLAYER CHOSEN MUST NECESSARILY BE AMONG THE POSSIBLE PLAYERS CONSIDERED IN THE FOOTBALL SCOUT REPORT.
                INDICATE the player chosen at the end of the REPORT.

                DATA:
                ------------------------------------
                {content}
                ------------------------------------ 

                TEAM REQUIREMENTS:
                Style of play: {style_t}
                Player type required: {type_player}
                Preferred league: {league}
                Key ability: {ability}
                Ideal formation: {formation}

                EXAMPLE TO FOLLOW:
                ### Report
                After a detailed analysis of the data, we have identified candidates who best meet the requirements of your team. Below, we present three potential candidates:

                ##### Three potential candidates:

                **[Player X]**: Highlights strengths and addresses weaknesses based on data on the essential attributes for a player in his specific age group.
                **[Player Y]**: Highlights strengths and addresses weaknesses based on data regarding the attributes a player must necessarily possess in his specific age group.
                **[Player Z]**: Highlighting strengths and addressing weaknesses based on attribute data that a player must necessarily possess in his specific age group.
                
                [Provide the reasons for choosing the recommended player over the others].
                
                The recommended player:
                [INDICATE Name of player recommended].
                """

            human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

            st.caption("Text generated by Scouter AI:")
            # Creazione di un messaggio di caricamento
            with st.spinner("Generating text. Please wait..."):
                llm = ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo") 
                chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
                result = llm(
                    chat_prompt.format_prompt(
                        player=choice, 
                        content=similar_players_df, 
                        style_t=game_styles[selected_game_style], 
                        type_player=player_experience[selected_player_experience], 
                        league=leagues[selected_league], 
                        ability=selected_player_skills, 
                        formation=selected_formation                    
                    ).to_messages()
                )

            # Extract the last item in the list
            st.markdown(result.content)

            # Use a regular expression to find the name after "The recommended player: "
            pattern = r"The recommended player:\s*([^:]+)"

            # find the correspondence in the entire text
            matches = re.findall(pattern, result.content, re.IGNORECASE)
            if matches:
                ultimo_nome = matches[0].rstrip('.')  # remove extra dot

    ####### Analytics of the recommended player ##############
    if form:  
        if matches:
            st.subheader("🌟 The features of the recommended player:")
            url_player = bing_image_urls(ultimo_nome+ " "+df_player.loc[df_player['Player'] == ultimo_nome, 'Squad'].iloc[0]+" 2023", limit=1, )[0]

            with st.expander("Selected Player", expanded=True):

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.subheader(ultimo_nome)
                        st.image(url_player, width=356)

                    with col2:
                        st.caption("📄 Information of Player")
                        col_1, col_2, col_3 = st.columns(3)

                        with col_1:
                            st.metric("Nation", df_player.loc[df_player['Player'] == ultimo_nome, 'Nation'].iloc[0], None)
                            st.metric("Position", df_player.loc[df_player['Player'] == ultimo_nome, 'Pos'].iloc[0], None)

                        with col_2:
                            st.metric("Born",df_player.loc[df_player['Player'] == ultimo_nome, 'Born'].iloc[0],None)
                            st.metric("Match Played",df_player.loc[df_player['Player'] == ultimo_nome, 'Playing_Time_MP'].iloc[0],None, help="In 2022/2023")

                        with col_3:
                            st.metric("Age",df_player.loc[df_player['Player'] == ultimo_nome, 'Age'].iloc[0],None)

                        st.metric(f"🏆 League: {df_player.loc[df_player['Player'] == ultimo_nome, 'Comp'].iloc[0]}",df_player.loc[df_player['Player'] == ultimo_nome, 'Squad'].iloc[0],None, help="In 2022/2023")


                    with col3:
                        st.caption("⚽ Information target of Player")
                        #GK
                        if df_player.loc[df_player['Player'] == ultimo_nome, 'Pos'].iloc[0] == "GK":
                            col_1, col_2 = st.columns(2)

                            with col_1:
                                st.metric("Saves", df_player.loc[df_player['Player'] == ultimo_nome, 'Performance_Saves'].iloc[0], None)
                                st.metric("Clean Sheet", df_player.loc[df_player['Player'] == ultimo_nome, 'Performance_CS'].iloc[0], None)

                            with col_2:
                                st.metric("Goals Against",df_player.loc[df_player['Player'] == ultimo_nome, 'Performance_GA'].iloc[0],None)
                                st.metric("ShoTA",df_player.loc[df_player['Player'] == ultimo_nome, 'Performance_SoTA'].iloc[0],None)
                        
                        # DF
                        if df_player.loc[df_player['Player'] == ultimo_nome, 'Pos'].iloc[0] == "DF" or df_player.loc[df_player['Player'] == ultimo_nome, 'Pos'].iloc[0] == "DF,MF" or df_player.loc[df_player['Player'] == ultimo_nome, 'Pos'].iloc[0] == "DF,FW":
                            col_1, col_2, col_3 = st.columns(3)

                            with col_1:
                                st.metric("Assist", df_player.loc[df_player['Player'] == ultimo_nome, 'Performance_Ast'].iloc[0], None)
                                st.metric("Goals", df_player.loc[df_player['Player'] == ultimo_nome, 'Performance_Gls'].iloc[0], None)

                            with col_2:
                                st.metric("Aerial Duel",df_player.loc[df_player['Player'] == ultimo_nome, 'Aerial_Duels_Won'].iloc[0],None)
                                st.metric("Tackle",df_player.loc[df_player['Player'] == ultimo_nome, 'Tackles_TklW'].iloc[0],None, help="In 2022/2023")

                            with col_3:
                                st.metric("Interception",df_player.loc[df_player['Player'] == ultimo_nome, 'Int'].iloc[0],None)
                                st.metric("Key Passage",df_player.loc[df_player['Player'] == ultimo_nome, 'KP'].iloc[0],None)

                        # MF
                        if df_player.loc[df_player['Player'] == ultimo_nome, 'Pos'].iloc[0] == "MF" or df_player.loc[df_player['Player'] == ultimo_nome, 'Pos'].iloc[0] == "MF,DF" or df_player.loc[df_player['Player'] == ultimo_nome, 'Pos'].iloc[0] == "MF,FW":
                            col_1, col_2, col_3 = st.columns(3)

                            with col_1:
                                st.metric("Assist", df_player.loc[df_player['Player'] == ultimo_nome, 'Performance_Ast'].iloc[0], None)
                                st.metric("Goals", df_player.loc[df_player['Player'] == ultimo_nome, 'Performance_Gls'].iloc[0], None)
                                st.metric("Aerial Duel",df_player.loc[df_player['Player'] == ultimo_nome, 'Aerial_Duels_Won'].iloc[0],None)

                            with col_2:
                                st.metric("GCA",df_player.loc[df_player['Player'] == ultimo_nome, 'GCA_GCA'].iloc[0],None)
                                st.metric("Progressive PrgP",df_player.loc[df_player['Player'] == ultimo_nome, 'Progression_PrgP'].iloc[0],None, help="In 2022/2023")

                            with col_3:
                                st.metric("SCA",df_player.loc[df_player['Player'] == ultimo_nome, 'SCA_SCA'].iloc[0],None)
                                st.metric("Key Passage",df_player.loc[df_player['Player'] == ultimo_nome, 'KP'].iloc[0],None)
                        
                        # FW
                        if df_player.loc[df_player['Player'] == ultimo_nome, 'Pos'].iloc[0] == "FW" or df_player.loc[df_player['Player'] == ultimo_nome, 'Pos'].iloc[0] == "FW,MF" or df_player.loc[df_player['Player'] == ultimo_nome, 'Pos'].iloc[0] == "FW,DF":
                            col_1, col_2, col_3 = st.columns(3) 

                            with col_1:
                                st.metric("Assist", df_player.loc[df_player['Player'] == ultimo_nome, 'Performance_Ast'].iloc[0], None)
                                st.metric("Goals", df_player.loc[df_player['Player'] == ultimo_nome, 'Performance_Gls'].iloc[0], None)
                                st.metric("Aerial Duel",df_player.loc[df_player['Player'] == ultimo_nome, 'Aerial_Duels_Won'].iloc[0],None)

                            with col_2:
                                st.metric("SCA",df_player.loc[df_player['Player'] == ultimo_nome, 'SCA_SCA'].iloc[0],None)
                                st.metric("xG",df_player.loc[df_player['Player'] == ultimo_nome, 'Expected_xG'].iloc[0],None)
                                st.metric("xAG",df_player.loc[df_player['Player'] == ultimo_nome, 'Expected_xAG'].iloc[0],None, help="In 2022/2023")
                                

                            with col_3:
                                st.metric("GCA",df_player.loc[df_player['Player'] == ultimo_nome, 'GCA_GCA'].iloc[0],None)
                                st.metric("Key Passage",df_player.loc[df_player['Player'] == ultimo_nome, 'KP'].iloc[0],None)
                    
            