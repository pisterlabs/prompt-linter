import streamlit as st
import json
import openai

# ------------------------------------------------------------------------------
# button example


def button_counter():
    st.title('Counter Example')
    count = 0

    increment = st.button('Increment')
    if increment:
        count += 1

    st.write('Count = ', count)

# ------------------------------------------------------------------------------
# loading data examples, caching


def bad_example_load_data():
    with open('data.json', 'r') as f:
        data = json.load(f)
    return data


@st.cache_data
def load_data():
    with open('data.json', 'r') as f:
        data = json.load(f)
    return data

# pokemon data from https://github.com/ezeparziale/pokemon-streamlit/blob/main/app/data/data.json

# ------------------------------------------------------------------------------
# select favourite pokemon


def select_favourite_pokemon(data):

    # Dropdown to select Pokémon
    pokemon_name = st.selectbox("Choose your favorite Pokémon", [v['name'].capitalize() for v in data.values()])

    name_to_id = {v['name'].capitalize(): k for k, v in data.items()}
    pokemon = data[name_to_id[pokemon_name]]

    # Display Pokémon image
    st.image(pokemon['img'], width=300)


def select_favourite_pokemon_centered(data):

    # Dropdown to select Pokémon
    pokemon_name = st.selectbox("Choose your favorite Pokémon", [v['name'].capitalize() for v in data.values()])

    name_to_id = {v['name'].capitalize(): k for k, v in data.items()}
    pokemon = data[name_to_id[pokemon_name]]

    # Display Pokémon image
    _, col2, _ = st.columns([1, 1, 2])  # you can choose colum sizes by passing a list of sizes
    with col2:
        st.image(pokemon['img'], width=300)


# ------------------------------------------------------------------------------
# create multiselect box with favorite pokemon types

def select_favourite_pokemon_types(data):
    # Multiselect to select Pokémon types

    pokemon_types = set()
    for pokemon in data.values():
        for type_ in pokemon['types']:
            pokemon_types.add(type_)
    pokemon_types = list(pokemon_types)

    TYPE_EMOJIS = {
        "grass": "🌱",
        "poison": "☠️",
        "fire": "🔥",
        "water": "💧",
        "electric": "⚡",
        "flying": "🕊️",
        "bug": "🐞",
        "normal": "🙂",
        "fairy": "🧚",
        "psychic": "🔮",
        "fighting": "🥊",
        "rock": "🪨",
        "ground": "🌍",
        "steel": "🔩",
        "ice": "❄️",
        "ghost": "👻",
        "dragon": "🐉",
        "dark": "🌑"
    }

    # make it fun, add emojis
    pokemon_types = [f"{TYPE_EMOJIS[t]} {t.capitalize()}" for t in pokemon_types]

    favorite_types = st.multiselect("Choose your favorite Pokémon types", pokemon_types)

# ------------------------------------------------------------------------------
# create form


def create_form(data):
    with st.form(key="hpi_survey"):
        st.title("HPI Survey")

        # Interaction Frequency
        st.radio("How often do you engage with Pokémon on a weekly basis?",
                                    ["Daily", "Several times a week", "Once a week", "Rarely", "Never"])

        # Emotional Connection
        st.selectbox("Do you feel a strong emotional connection to any specific Pokémon?",
                                         [v['name'].capitalize() for v in data.values()])
        st.text_input("Why do you feel emotionally connected to this Pokémon?")

        # Influence on Mood
        st.select_slider("Has interacting with a particular Pokémon affected your mood?", [ "Negatively",  "No effect", "Positively"])

        # Virtual Interfaces
        st.radio(
            "Which virtual interface or platform do you use most frequently to interact with Pokémon?",
            ["Pokémon GO", "Pokémon Sword and Shield", "Pokémon Trading Card Game Online", "Other"])

        # Ethical Concerns
        st.text_area(
            "Do you have any ethical concerns about the way Pokémon are treated in virtual interfaces?")

        submitted = st.form_submit_button("Submit")
        if submitted:
            st.success("✅ We received your submission!")


# ------------------------------------------------------------------------------
# chat with pokemon
def pokemon_prompt(name):
        return (f"You are {name}, a friendly Pokémon. A curious human approaches you and wants to have a chat."
                f"Be {name}. Answer the human's questions. Be friendly. Tell them about yourself. Tell them about your experience with being a Pokémon.")


def chat_with_pokemon_bad(data):

    st.divider()

    # Dropdown to select Pokémon, with which pokemon would you like to chat?
    pokemon_name = st.selectbox("Choose Pokémon to chat with!", [v['name'].capitalize() for v in data.values()])

    name_to_id = {v['name'].capitalize(): k for k, v in data.items()}
    pokemon = data[name_to_id[pokemon_name]]

    st.title(f"💬 Chat with {pokemon_name}")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "system", "content": pokemon_prompt(pokemon_name)},
                                        {"role": "assistant", "content": f"Hi! I am {pokemon_name}!"}]

    for msg in st.session_state.messages:
        if msg["role"] == "assistant":
            st.chat_message(msg["role"], avatar=pokemon['img']).write(msg["content"])
        elif msg["role"] == "user":
            st.chat_message(msg["role"]).write(msg["content"], )

    if prompt := st.chat_input():
        openai.api_key = st.secrets.OPENAI_API_KEY
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
        msg = response.choices[0].message
        st.session_state.messages.append(msg)
        st.chat_message("assistant", avatar=pokemon['img']).write(msg.content)


def remove_chat_history_callback():
    if "messages" in st.session_state:
        del st.session_state["messages"]


def chat_with_pokemon(data):

    st.divider()

    # Dropdown to select Pokémon, with which pokemon would you like to chat?
    pokemon_name = st.selectbox("Choose Pokémon to chat with!", [v['name'].capitalize() for v in data.values()],
                                on_change=remove_chat_history_callback)

    name_to_id = {v['name'].capitalize(): k for k, v in data.items()}
    pokemon = data[name_to_id[pokemon_name]]

    st.title(f"💬 Chat with {pokemon_name}")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "system", "content": pokemon_prompt(pokemon_name)},
                                        {"role": "assistant", "content": f"Hi! I am {pokemon_name}!"}]

    for msg in st.session_state.messages:
        if msg["role"] == "assistant":
            st.chat_message(msg["role"], avatar=pokemon['img']).write(msg["content"])
        elif msg["role"] == "user":
            st.chat_message(msg["role"]).write(msg["content"], )

    if prompt := st.chat_input():
        openai.api_key = st.secrets.OPENAI_API_KEY
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
        msg = response.choices[0].message
        st.session_state.messages.append(msg)
        st.chat_message("assistant", avatar=pokemon['img']).write(msg.content)

# ------------------------------------------------------------------------------
