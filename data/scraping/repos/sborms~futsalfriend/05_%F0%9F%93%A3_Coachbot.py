import time

import streamlit as st
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from openai.error import AuthenticationError

st.set_page_config(page_title="Coachbot", page_icon="📣", layout="wide")

import queries

@st.cache_resource(show_spinner=False, ttl=1800)
def load_chain(input_openai_api_key, context=""):
    """Configures a conversational chain for answering user questions."""
    # load OpenAI's language model
    llm = ChatOpenAI(
        temperature=0.5, model="gpt-3.5-turbo", openai_api_key=input_openai_api_key
    )

    # create chat history memory
    memory = ConversationBufferWindowMemory(
        k=3, memory_key="history", ai_prefix="Coach"
    )

    # create prompt
    prefix = f"""
    You are an AI assistant that provides advice to futsal teams.
    Futsal is played 5 against 5 on a small indoor field.
    You are given the following information and a question.
    Provide a conversational answer. Be witty but concise. Don't make up an answer.
    If the question is not about futsal, inform that you are tuned
    to only answer questions about futsal.

    Below is relevant information about the team and competition.
    {context}
    """
    template = (
        prefix
        + """History:

    \n{history}\n
    
    Question: {input}
    """
    )

    # define prompt template
    prompt = PromptTemplate(input_variables=["history", "input"], template=template)

    # Create the conversational chain
    chain = ConversationChain(prompt=prompt, llm=llm, memory=memory, verbose=True)

    return chain


def prepare_prompt_team_context(dict_info):
    """Prepares a string with relevant team information as context for the prompt."""
    context = "\n"
    for title, df in dict_info.items():
        context += title + ":\n" + df.to_string(index=False) + "\n\n"
    return context


def lets_chat():
    st.cache_data.clear()
    st.cache_resource.clear()

    st.session_state["lets_chat"] = True


##################
########## UI   ##
##################

avatar_ai = "assets/capi.png"
avatar_player = "assets/player.svg"

st.title("💬 Coachbot")
st.markdown("### Seek advice from an AI futsal coach")

# initialize session state
if "team" not in st.session_state:
    st.session_state["team"] = "ZVC Copains"
if "lets_chat" not in st.session_state:
    st.session_state["lets_chat"] = False

# ask for team first
if not st.session_state["lets_chat"]:
    st.cache_data.clear()

    teams_all = queries.query_list_teams()["team"].tolist()

    col1, _, _ = st.columns(3)
    team = col1.selectbox(
        "First tell me what team you play for",
        options=teams_all,
        index=teams_all.index(st.session_state["team"]),
    )
    st.session_state["team"] = team

    st.button("Let's chat!", on_click=lets_chat)

# initialize navbar and chat window
if st.session_state["lets_chat"]:
    with st.sidebar:
        with st.container():
            st.success(
                "The chatbot uses OpenAI's **GPT-3.5 Turbo** language model.",
                icon="ℹ️",
            )

            bot_type = st.radio(
                "Choose a type of bot",
                ["Basic", ":red[Advanced]"],
                captions=[
                    "Free until it lasts but slower.",
                    "Paid but faster and more capable.",
                ],
                index=0,
            )

            if "Basic" in bot_type:
                input_openai_api_key = st.secrets["openai"]["api_key_free"]
            elif "Advanced" in bot_type:
                st.info(
                    """
                    Enter your OpenAI API key (we won't expose it!) to use your own account.
                    For pricing info see [here](https://openai.com/pricing#language-models).
                    """
                )
                input_openai_api_key = st.text_input(
                    "Paste your key here:",
                    type="password",
                    placeholder="sk-...",
                    value="sk-...",
                )

    st.markdown(
        "*You can **try** to write in any language other than English. "
        "Always take what the bot says with a grain of salt.* 😊"
    )

    # get relevant information to add as context to prompt
    team = st.session_state["team"]

    df_schedule = queries.query_schedule(team=team)

    df_stats_players = queries.query_stats_players(team=team)

    oponnent_1 = [who for who in df_schedule.loc[0][1:].tolist() if who != team][0]
    df_stats_players_oponnent_1 = queries.query_stats_players(team=oponnent_1)

    df_standings = queries.query_standings(team=team)

    dict_info = {
        "Competition standings": df_standings,
        "Schedule": df_schedule,
        "Player statistics": df_stats_players,
        "Player statistics next opponent": df_stats_players_oponnent_1,
    }
    context = prepare_prompt_team_context(dict_info)

    # configure chain
    chain = load_chain(input_openai_api_key, context=context)

    # initialize chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": f"Cool, {team} is a great team! How can I help?",
            }
        ]

    # display chat messages from history on app rerun
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            with st.chat_message(message["role"], avatar=avatar_ai):
                st.markdown(message["content"])
        else:
            with st.chat_message(message["role"], avatar=avatar_player):
                st.markdown(message["content"])

    if query := st.chat_input("Talk to me..."):
        # display user message
        with st.chat_message("user", avatar=avatar_player):
            st.markdown(query)

        # add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})

        # display chatbot message
        with st.chat_message("assistant", avatar=avatar_ai):
            # send user's question to chain
            try:
                result = chain({"input": query})
            except AuthenticationError:
                st.warning("Your API key is invalid or expired...")
                st.stop()

            response = result["response"].replace("Coach: ", "")

            message_placeholder, full_response = st.empty(), ""
            for chunk in response.split():
                full_response += chunk + " "
                time.sleep(0.05)  # simulate stream of response with milliseconds delay
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        # add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
