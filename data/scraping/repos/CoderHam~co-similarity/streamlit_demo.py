import pandas as pd
import numpy as np
import difflib
import streamlit as st
import streamlit.components.v1 as components

import cohere

bg_df = pd.read_csv("bgg_2000.csv")
name_list = bg_df.loc[:, 'name'].to_list()
thumbnail_list = bg_df.loc[:, 'thumbnail'].to_list()
url_list = bg_df.loc[:, 'url'].to_list()
co = cohere.Client(st.secrets["cohere_api_token"])

def info_from_ids(game_ranks):
    game_names = [name_list[i] for i in game_ranks]
    img_urls = [thumbnail_list[i] for i in game_ranks]
    game_urls = [url_list[i] for i in game_ranks]
    return game_names, img_urls, game_urls

def get_html_code(game_names, img_urls, game_urls):
    html_code ="""
<style>
    img {
        float: left;
        padding: 8px;
    }
</style>
    """
    for game_name, img_url, game_url in zip(game_names, img_urls, game_urls):
        html_code += """
<a href="%s"><img src="%s" style="width: 300px; height: 300px;" title="%s"></a>
        """ % (game_url, img_url, game_name)
    return html_code

def result_box(game_names, img_urls, game_urls):
    st.markdown(get_html_code(game_names, img_urls, game_urls), unsafe_allow_html=True)

def getSimilarFromGame():
    import similarity_search
    name_query = st.session_state.game_name
    st.session_state.game_name = ''

    game_index = 0
    print(name_query)
    if name_query not in name_list:
        name_list_lower = [name.lower() for name in name_list]
        name_query_lower = name_query.lower()
        closest_names = difflib.get_close_matches(name_query_lower, name_list_lower)
        game_index = name_list_lower.index(closest_names[0])
        print("Did you mean :", name_list[game_index], "?")
    else:
        game_index = name_list.index(name_query)

    results = similarity_search.get_similar_from_game(game_index, k=9)
    query_name, query_url, query_url = info_from_ids([game_index])
    game_names, img_urls, game_urls = info_from_ids(results)
    result_box(game_names, img_urls, game_urls)

def getSimilarFromDescription():
    import similarity_search
    game_description = st.session_state.game_description
    st.session_state.game_description = ''

    sentence = "railway build across United States"
    query_embeds = co.embed(model='small', texts=[game_description]).embeddings
    results = similarity_search.get_similar_from_embeds(query_embeds, 9)
    game_names, img_urls, game_urls = info_from_ids(results)
    result_box(game_names, img_urls, game_urls)

if __name__ == "__main__":
    st.set_page_config(
        page_title="Board Game Similarity Search",
        page_icon="https://vectorified.com/images/board-game-icon-4.png",
        layout="wide",
    )

    name_query = st.text_input('Board Game Name', '', key="game_name", on_change=getSimilarFromGame, placeholder="Enter your favorite board game...")
    description = st.text_input('Board Game Description', '', key="game_description", on_change=getSimilarFromDescription, placeholder="Enter a board game description...")
