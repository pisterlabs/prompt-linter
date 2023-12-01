# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import streamlit as st
from PIL import Image
from streamlit.logger import get_logger
import openai

LOGGER = get_logger(__name__)


try: 
    openai_api_key = os.getenv("OPENAI_API_KEY")
except: 
    openai_api_key = st.secrets["OPENAI_API_KEY"]


image = Image.open('images/producttoer.jpeg')
ENCODINGS = 'cl100k_base'

def run():
    st.set_page_config(
            page_title=":genie:Berend-Botje Skills",
            page_icon=" :genie:👋",
            layout="wide",
            initial_sidebar_state="collapsed",
            menu_items={
                'Get Help': 'https://mboscrum.com/mbowoordle',
                'Report a bug':'https://mboscrum.com/mbowoordle',
                'About': "# Berend in development -  een *extremely* cool app!"
            }
        )

    col1, col2 = st.columns(2)
    with col1:
        st.header(":genie::infinity: Welkom bij Berend-Botje")     
        st.markdown("""
        ##### Berend-Botje is een slimme AI assistent met skills die perfect aansluiten bij het *Smart Working principle* ####""")
        st.markdown("""
        ###### Berend-Botje Basis:male_mage:, een soort van ChatGPT kloon, staat altijd voor je klaar om snel je vragen te beantwoorden. Heb je behoefte aan hulp bij een specifieke taak, dan vraag je Berend om de bijpassende Skills voor je in te pluggen. 
        **Jij kiest dus op basis van je klus de bijpassende Berend Bot.**  
        :rotating_light: Belangrijk voordeel van het gebruik van Berend-Botje is dat al jouw informatie binnen jouw omgeving blijft!  *Nadat een sessie wordt afgesloren blijft er dus geen informatie achter die door ons noch door derden gebruikt kan worden!*
        ------------------------------------
        >> De skills zijn **Powered By OpenAI models**
        """ 
        )

    with col2:
        st.image(image, caption=None, use_column_width=True, clamp=True, channels="RGB", output_format="auto")
        st.markdown(""" 
        ##### Voorbeelden
        **1. [De Lesplanner](Lesplan_Demo)**
        **2. [De Notulist](Mapping_Demo)**
        **3. [De Dataanalist](DataFrame_Demo)**
        **4. [De Datavormgever](Plotting_Demo)**
        **5. [De Chatbot](Chat_Demo)**
        **6. [De Samenvatter](Samenvatter_Demo)**
        **9. [Berend Broodjes](Broodje_Berend_Demo)**
        """
                   )
    
    st.markdown("""
    :angel: :gray[ *Disclaimer Aan het gebruik, of resulaten van Berend-Botje Skills kunnen geen rechten worden verleend. Noch zijn wij aansprakelijk voor enig gevolg van dit gebruik. Bedenk dat de voorbeelden die hier getoond worden nog in een premature fase verkeren: het is werk onder constructie...* ]
    """
    )


if __name__ == "__main__":
    run()
