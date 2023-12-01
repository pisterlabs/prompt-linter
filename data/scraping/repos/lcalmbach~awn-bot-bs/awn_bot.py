import streamlit as st
import pandas as pd
import time
import json
from texts import texts
from const import (
    ADR_FILE,
    APARTMENT_FILE,
    SYSTEM_PROMPT,
    user_questions,
    assistant_responses,
    LLM_RETRIES,
    SLEEP_TIME_AFTER_ERROR,
)
from tools import show_table, get_var
from openai import OpenAI
from enum import Enum
import difflib
import folium
from streamlit_folium import folium_static


class Status(Enum):
    street = 1
    housenumber = 2
    plz = 3
    location = 4
    egid = 5
    floor = 6
    apartment = 7


class AwnBot:
    def __init__(self, mode: str = "interactive"):
        self.mode = mode
        self.addresses = pd.read_parquet(ADR_FILE)
        self.apartments = pd.read_parquet(APARTMENT_FILE)
        self.streets = list(self.addresses["strname"].unique())
        self.locations = list(self.addresses["dplzname"].unique())
        self.streets.sort()
        self.status = Status.street
        self.model = "gpt-3.5-turbo"
        self.temperature = 0
        self.max_tokens = 1000
        # counts how often a status was reached, if a status is reached multiple times, the assistant will give more information
        self.status_count = self.args = {
            Status.street.name: 0,
            Status.housenumber.name: 0,
            Status.plz.name: 0,
            Status.location.name: 0,
            Status.egid.name: 0,
            Status.floor.name: 0,
            Status.apartment.name: 0
        }
        # These are the slots that are filled by the users answers
        self.args = {
            Status.street.name: "",
            Status.housenumber.name: "",
            Status.plz.name: "",
            Status.location.name: "",
            Status.egid.name: "",
            Status.floor.name: "",
            Status.apartment.name: "",
        }

    def show_info(self):
        st.subheader("🔎AWN-Finder")
        with st.expander("Informationen zur App"):
            st.markdown(f"{texts['info']}")

    def show_map(self, df):
        """
        Display a map with markers for each location in the given DataFrame.

        Parameters:
        - df (pandas.DataFrame): The DataFrame containing location data with latitude and longitude columns.

        Returns:
        None
        """
        map_center = [df["latitude"].mean(), df["longitude"].mean()]
        map = folium.Map(location=map_center, zoom_start=18)
        tool_tip = f'''{self.street} {self.housenumber}\n
    {self.plz} {self.location}\n\n EGID: {self.egid}'''
        for index, row in df.iterrows():
            folium.Marker(
                [row["latitude"], row["longitude"]],
                popup=tool_tip,
            ).add_to(map)
        folium_static(map)

    def show_interactive(self):
        st.markdown("**Gib bitte deine Adresse ein:**")
        self.egid = 0
        cols = st.columns(2)
        with cols[0]:
            options_streets = ["<Wähle eine Strasse>"] + self.streets
            self.street = st.selectbox("Strasse", options=options_streets)
        with cols[1]:
            if options_streets.index(self.street) == 0:
                housenumbers = self.addresses["deinr"].unique()
            else:
                housenumbers = self.addresses[self.addresses["strname"] == self.street][
                    "deinr"
                ].unique()
            housenumbers = [x for x in housenumbers if x is not None]
            housenumbers.sort()

            hrn_options = ["<Wähle eine Hausnummer>"] + housenumbers
            self.housenumber = st.selectbox(
                "Hausnummer",
                options=hrn_options,
                disabled=self.street == "<Wähle eine Strasse>",
            )

        with cols[0]:
            if hrn_options.index(self.housenumber) == 0:
                plz_options = self.addresses["dplz4"].unique().sort()
                plz_options = ["Wähle eine Postleitzahl"] + [plz_options]
            else:
                plz_options = self.addresses[
                    (self.addresses["strname"] == self.street)
                    & (self.addresses["deinr"] == self.housenumber)
                ]["dplz4"].unique()
                plz_options = [str(x) for x in plz_options if x is not None]
            self.plz = st.selectbox("Postleitzahl", options=plz_options)
        record = self.addresses[
            (self.addresses["strname"] == self.street)
            & (self.addresses["deinr"] == self.housenumber)
        ][["dplzname", "egid", "latitude", "longitude"]].drop_duplicates()

        with cols[1]:
            if self.plz == "Wähle eine Postleitzahl":
                ort_options = list(self.addresses["dplzname"])
                ort_options.sort()
            else:
                ort_options = record["dplzname"]
                self.egid = record.iloc[0]["egid"]
            self.location = st.selectbox("Ort:", ort_options, disabled=True)
        if self.egid > 0:
            self.show_map(record)
        # st.markdown('**Gebäude**')
        # st.markdown(f'Egid: {self.egid}')
        cols = st.columns(2)
        options_floors = self.apartments[self.apartments["egid"] == self.egid][
            "wstwk_decoded"
        ].unique()
        options_floors.sort()
        with cols[0]:
            self.floor = st.selectbox("Stockwerk", options=options_floors)
        contact_email = "vitus.thali@bs.ch"
        contact = "Vitus Thali"
        st.markdown(    
            f'Wähle deine Wohnung aus. Bei mehreren Wohnungen auf dem gleichen Stockwerk kannst kannst die Zimmerzahl oder Grösse der Wohnung verwenden, um deine Wohnng zu erkennen. Bist du unsicher, dann wende dich bitte an <a href="mailto:{contact_email}">{contact}</a> (Statistisches Amt Basel Stadt).',
            unsafe_allow_html=True,
        )
        df = self.apartments[
            (self.apartments["egid"] == self.egid)
            & (self.apartments["wstwk_decoded"] == self.floor)
        ]
        fields = ["egid", "whgnr", "wstwk_decoded", "wbez", "warea", "wazim", "ewid"]
        df = df[fields]
        df = df.sort_values("wstwk_decoded")
        df.columns = [
            "EGID",
            "AWN",
            "Stockwerk",
            "Info",
            "Fläche",
            "Anz. Zimmer",
            "EWID",
        ]
        settings = {"height": 200}
        response = show_table(df, [], settings)
        if response:
            st.subheader(f"Deine Wohnungsnummer: {response['AWN']}")

    def get_completion(self, messages):
        """Generates a response using the OpenAI ChatCompletion API based on
        the given text.

        Args:
            text (str): The user's input.

        Returns:
            str: The generated response.

        Raises:
            None
        """
        client = OpenAI(
            api_key=get_var("OPENAI_API_KEY"),
        )
        retries = LLM_RETRIES
        while retries > 0:
            try:
                completion = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                tokens = [
                    completion.usage.completion_tokens,
                    completion.usage.prompt_tokens,
                ]
                response = completion.choices[0].message.content
                print(response)
                return response, tokens
            except Exception as err:
                st.error(f"OpenAIError {err}")
                retries -= 1
                time.sleep(SLEEP_TIME_AFTER_ERROR)
        return "", []

    def get_valid_street_name(self, street_name: str) -> str:
        # Normalize street names (e.g., lowercase)
        street_name = street_name.lower().replace(".", "")
        # Replace common abbreviations
        street_name = street_name.replace("str", "strasse")
        # Find close matches
        close_matches = difflib.get_close_matches(
            street_name, self.streets, n=1, cutoff=0.8
        )
        if len(close_matches) > 0:
            return close_matches[0]

    def get_valid_locations(self, location: str) -> str:
        close_matches = difflib.get_close_matches(
            location, self.locations, n=1, cutoff=0.8
        )
        if len(close_matches) > 0:
            return close_matches[0]

    def get_valid_housenumber(self, housenumber: str):
        housenumber = housenumber.lower().replace(".", "")
        df = self.addresses[
            (self.addresses["strname"].str.lower() == self.args["street"].lower())
            & (self.addresses["deinr"] == housenumber)
        ]
        if len(df) > 0:
            return housenumber, df
        else:
            return None, None

    def address_text(self):
        return f'{self.args["street"]} {self.args["housenumber"]}, {self.args["plz"]} {self.args["location"]} (EGID Gebäude: {self.args["egid"]})'

    def increment_status(self):
        self.status = Status(self.status.value + 1)
        print(self.status)

    def get_status(self, response):
        status = self.status.value
        self.status_count[Status.street.name] += 1
        for key, value in response.items():
            if key == Status.street.name and value > "" and status < Status.housenumber.value:
                
                street = self.get_valid_street_name(value)
                if street:
                    self.args["street"] = street
                    self.increment_status()
            if key == "housenumber" and value > "" and status < Status.plz.value:
                hnr, candidates = self.get_valid_housenumber(value)
                if hnr is None:
                    self.status = Status.housenumber
                elif isinstance(hnr, list):
                    # get the list of hrn and let the assisten say htat the hnr does not exist and the user
                    # has to choose one of the options
                    ...
                else:
                    self.args["housenumber"] = hnr
                    self.increment_status()
                    plz = candidates["dplz4"].unique()
                    if len(plz) == 1:
                        self.args["plz"] = plz[0]
                        self.increment_status()
                        self.args["location"] = candidates["dplzname"].unique()[0]
                        self.increment_status()
                        self.args["egid"] = candidates.iloc[0]["egid"]
                        self.increment_status()
            if key == "plz" and value > "" and status < Status.floor.value:
                ...
            if key == "location" and value > "" and status < Status.egid.value:
                ...
            if key == "floor" and value > "" and status < Status.egid.value:
                ...

    def get_floors(self):
        return self.apartments[self.apartments["egid"] == self.args["egid"]][
            "wstwk_decoded"
        ].unique()
    
    def get_apartments_on_floor(self):
        apartments_on_floor = self.apartments[
            (self.apartments["egid"] == self.args["egid"]) &
            (self.apartments["wstwk_decoded"] == self.args["floor"])
        ]
        return apartments_on_floor

    def get_records(self):
        addresses = self.addresses
        apartments = self.apartments
        # st.write(self.args)
        if self.args["street"] > "":
            addresses = addresses[
                (addresses["strname"] == self.args['street'])
            ]
        if self.args["housenumber"] > "":
            addresses = addresses[
                (addresses["deinr"] == self.args['housenumber'])
            ]
        if self.args["plz"] != "":
            addresses = addresses[
                (addresses["dplz4"] == self.args['plz'])
            ]
        if (len(addresses) > 0) & (self.args["location"] > ""):
            addresses = addresses[
                (addresses["dplzname"] == self.args['plz'])
            ]
        if len(addresses) == 1:
            self.args['egid'] = addresses.iloc[0]['egid']

        if self.args["egid"] != "":
            apartments = apartments[apartments["egid"] == self.args["egid"]]
            if self.args["floor"] > "":
                addresses = addresses[
                    (addresses["egid"] == self.egid)
                ]
        return addresses, apartments

    def get_address(self):
        return f'{self.args["street"]} {self.args["housenumber"]}, {self.args["plz"]} {self.args["location"]} (EGID Gebäude: {self.args["egid"]})'
    
    def format_response(self, response):
        includes = []
        for include in assistant_responses[self.status.name]['includes']:
            if include == 'adrs':
                includes.append(self.get_address())
            if include == 'floors':
                includes.append(self.get_floors())
        if includes:
            response = response.format(*includes)
        return response
                
    def show_bot(self):
        st.session_state.display_messages = []
        adrs, aps = None, None
        if "messages" not in st.session_state:
            st.session_state.messages = []
            st.session_state.display_messages = []
            st.session_state.messages.append(
                {"role": "system", "content": SYSTEM_PROMPT}
            )
            st.session_state.display_messages.append(
                {
                    "role": "assistant",
                    "content": "Guten Tag, Ich helfe dir gerne, deine AWN zu finden. Bitte gib unten deine Adresse ein",
                }
            )

        if prompt := st.chat_input('Beispiel: Blumenrain 34, 4058 Basel, 1. Stock'):
            st.session_state.display_messages.append({"role": "user", "content": prompt})
            user_question = user_questions[self.status.name].format(prompt)
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": user_question,
                }
            )
            response, tokens = self.get_completion(st.session_state.messages)
            try:
                address_obj = json.loads(response)
                self.get_status(address_obj)
                assistant_response = assistant_responses[self.status.name]['content']
                assistant_response = self.format_response(assistant_response)
                

            except Exception as err:
                print(err)
                assistant_response = response
            
            adrs, aps = self.get_records()
            st.session_state.display_messages.append({"role": "assistant", "content": assistant_response})
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})

        for msg in st.session_state.display_messages:
            if msg["role"] != "system":
                st.chat_message(msg["role"]).write(msg["content"])
        
        if st.button("Starte nochmals neu"):
            del st.session_state['display_messages']
            del st.session_state['messages']

    def show_ui(self):
        self.show_info()
        if self.mode == "form":
            self.show_interactive()
        elif self.mode == "bot":
            self.show_bot()
