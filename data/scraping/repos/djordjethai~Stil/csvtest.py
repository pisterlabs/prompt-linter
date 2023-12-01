import streamlit as st
from langchain.agents import AgentType
from langchain.agents import create_csv_agent
from langchain.chat_models import ChatOpenAI
from myfunc.mojafunkcija import init_cond_llm
import io

st.subheader("Testiranje modela na osnovu csv fajla")
st.caption("Ver. 21.10.23")
st.divider()

model, temp = init_cond_llm()
uploaded_file = st.file_uploader(
    "Choose a CSV file", accept_multiple_files=False, type="csv", key="csv_key"
)
if uploaded_file is not None:
    with io.open(uploaded_file.name, "wb") as file:
        file.write(uploaded_file.getbuffer())
    with st.form("my_form"):
        upit = st.text_area("Sistem: ", value="Pisi iskljucivo na srpskom jeziku. ")
        posalji = st.form_submit_button("Posalji")

        if posalji:
            try:
                agent = create_csv_agent(
                    ChatOpenAI(temperature=temp, model=model),
                    "pravilnik.csv",
                    verbose=True,
                    agent_type=AgentType.OPENAI_FUNCTIONS,
                    handle_parsing_errors=True,
                )
            except Exception as e:
                st.write(f"Molim vas napisite pitanje drugacije, nisam razumeo... {e}")
            odgovor = agent.run(upit)
            st.write(odgovor)
