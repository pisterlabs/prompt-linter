import streamlit as st
import datetime
import pandas as pd
import streamlit_bd_cytoscapejs
import openai
import re

import streamlit as st
import pyrebase
import datetime
from streamlit_extras.switch_page_button import switch_page


if "auth_user" not in st.session_state or not st.session_state["auth_user"]:
  switch_page("login")

config = st.secrets
openai.api_key = config.open_api_key

config = st.secrets
firebase = pyrebase.initialize_app(config)
db = firebase.database()


uid = st.session_state["auth_user"]["localId"]

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191



st.set_page_config(layout="wide")

# Sample data
users = ["Theo", "Theo", "Theo", "Felix", "Felix", "Felix"]
journal_entries = [
    """Trying new fabrication method, namely to try new high-resolution mask for fabrication of photonic chip. 
    First attempt of using SU8 as a high-resolution mask. Made one initial attempt for developing SU8 using the following parameters: 
    Spin-Coating SU8 Soft Bake (3 min. at 60 degrees)
    Photolithography Exposure (lambda = 450 nm, P = 15 muW, t = 2 min.)
    Development (3 min. in SU8 developer)
    Looking at Chip under the microscope;
    This attempt Failed; The development step dissolved all the SU8 for an unknown reason so far. Will have to investigate tomorrow.""",
    """As yesterday's attempt of using SU8 as a high-resolution mask. Made further attempts for developing SU8 
    using the same steps as yesterday with various different parameters, all failed. The development step still dissolved all the SU8 for an unknown reason""",
    """Third attempt of using SU8 as a high-resolution mask. Realized that I skipped a step (the post-exposure step) that was flagged as optional in the SU8 manual. Introducing the step turned out to make the process work. The steps I followed for my working version:
    1. Spin-Coating SU8
    Soft Bake (3 min. at 60 degrees)
    Photolithography Exposure (lambda = 450 nm, P = 15 muW, t = 2 min.)
    Post-Exposure Bake (4 min.)
    Development (3 min. in SU8 developer)
    Looking at the chip under the microscope. This time I saw the SU8 on the chip after the development! --> method worked""",
    """For testing and designing my on chip cavity, I need to couple from fiber to on-chip waveguide on the room temperature setup.""",
    """trying to couple to the chip. Coupling not possible for unknown reason. Data jumps around weirdly. Tried methods that did not lead to success:
    - checking laser frequency
    - changing chip we know couples
    - checking pigtail connections
    - checking that laser exites through fiber with visible laser
    - turning off ventilation system to reduce noise""",
    """At the beginning of the day I still had no clue why it was not working; After a lot of discussions with group members, I
    finally figured out what the problem was: The detection diode was not calibrated correctly. So next time keep that in mind!"""
]
dates = [datetime.date(2021, 2, 1), datetime.date(2022, 2, 1), 
         datetime.date(2023, 2, 1), datetime.date(2021, 2, 1), datetime.date(2022, 2, 1),
         datetime.date(2023, 2, 1)]


# Create DataFrame
df = pd.DataFrame({'user': users, 'journal_entry': journal_entries, 'date': dates})
#ontological similarity

dates = df.date.unique()
users = df.user.unique()
dates = sorted(dates)


def entry(message, key, column):
    col1, col2 = column.columns([8,1])
    with col1:
      if "editing" in st.session_state and st.session_state["editing"] == key:
        edit_inp = column.text_area("Editing entry", value=message, key=key+"_edit")
        if edit_inp != message:
          db.child("journals").child(uid).child(key).update({"entry": edit_inp, "date": date})
          st.write("Updated entry", key)
          del st.session_state["editing"]
          st.rerun()
      else:
        column.markdown(message)
    with col2:
      if column.button("Edit", key=key):
        st.session_state["editing"] = key
        st.rerun()

def parse_database_entries(entries, uid):
    # Convert from pyrebase object to dict
    for i in range(len(entries)):
        key = entries[i].key()
        entries[i] = entries[i].val()
        entries[i]["key"] = key
        entries[i]["uid"] = uid
    return entries

def get_all_entries():
    entries = db.child("journals").child(uid).order_by_key().get().each()
    entries = parse_database_entries(entries, uid)

    if "collaborators" in st.session_state and st.session_state["collaborators"]:
        for col in st.session_state["collaborators"]:
            colab_entries = (
                db.child("journals").child(col.key()).order_by_key().get().each()
            )
            entries.extend(parse_database_entries(colab_entries, col.key()))
    return entries

entries = get_all_entries()
if entries:
    df = pd.DataFrame(entries)
    df["date"] = df["date"].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f').date())
    #st.write(df)
    
    for date in df["date"].unique():
        today_df = df[df["date"] == date]
        users = today_df["uid"].unique()
        # Create a column for each user
        columns = st.columns(len(users)+1)
        #columns[0].header("Date")
        columns[0].write(date)
        for i, column in enumerate(columns[1:]):
            user = users[i]
            with column:
                for entry in today_df.loc[today_df["uid"] == user,"entry"]:
                    #with st.chat_message(user):
                    st.markdown(entry)



if completion:
    st.write(completion["choices"][0]["message"]["content"])
    match = re.search(r'"Summary":\s+"(.*?)"', completion["choices"][0]["message"]["content"], re.DOTALL)
    if match:
        summary = match.group(1)
        st.write(summary)
    match = re.search(r'"Relevant messages": \[([\d, ]+)\]', completion["choices"][0]["message"]["content"])
    if match:
        messages_str = match.group(1)
        messages = [int(msg.strip()) for msg in messages_str.split(',')]
        st.write(messages)