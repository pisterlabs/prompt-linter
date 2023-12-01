import os
import streamlit as st

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import cohere

# Using the multilingual model to support names in different languages.
CONFIG_MODEL = 'embed-multilingual-v2.0'

# Read the Cohere API key from the environment.
co = cohere.Client(os.environ["COHERE_API_KEY"])

def main():
    st.markdown('# Identicons')

    st.markdown("""
    > Ref: [Wikipedia](https://en.wikipedia.org/wiki/Identicon) Don Park came up with the Identicon idea on January 18, 2007. In his words:
    >
    > I originally came up with this idea to be used as an easy means of visually distinguishing multiple units of information, anything that can be reduced to bits. It's not just IPs but also people, places, and things. IMHO, too much of the web what we read are textual or numeric information which are not easy to distinguish at a glance when they are jumbled up together. So I think adding visual identifiers will make the user experience much more enjoyable.

    ---
    """)

    left_column, right_column = st.columns(2)

    # Take inputs from the user in the left column.
    with left_column:
        st.text_input('Enter username:', key='username', value='bkowshik')
        length = st.slider('Select identicon size:', key='length', min_value=3, max_value=20, value=7)

    # Display the output identicon in the right column.
    with right_column:
        embedding = co.embed([st.session_state.username], model=CONFIG_MODEL).embeddings[0]
        embedding = embedding[:length ** 2]

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax = sns.heatmap(np.array(embedding).reshape(length, length), ax=ax, cbar=False, linewidths=0.01, square=True, xticklabels=False, yticklabels=False)
        st.pyplot(fig)

if __name__ == '__main__':
    main()
