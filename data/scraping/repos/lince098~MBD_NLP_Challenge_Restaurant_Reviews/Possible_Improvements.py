import streamlit as st
from challenge_functions import openai_interface as oi, general as g


st.sidebar.markdown("# Possible Improvements")
st.sidebar.markdown(
    "This functionality will suggest ways on how to improve the service based on the chosen customers reviews."
)

st.markdown("# Possible Improvements")


uploaded_file = st.file_uploader("Insert a csv file with TripAdvisor reviews.")

if uploaded_file:
    df = g.load_csv_from_file_input(uploaded_file)

    with st.form("my_form"):
        st.markdown("### You can select the rows you want to analyze")

        st.dataframe(df)

        selected_list = st.multiselect("Index", df.index.to_list())

        submitted = st.form_submit_button("Get Improvements")

    if submitted:
        improvements = oi.get_improvements(selected_list, df)

        if improvements:
            st.markdown(improvements)
        else:
            st.markdown(":red[No rows selected.]")
