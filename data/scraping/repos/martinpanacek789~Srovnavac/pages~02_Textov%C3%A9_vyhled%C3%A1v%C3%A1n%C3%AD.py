import streamlit as st

import re

import pandas as pd
import plotly.graph_objects as go
import openai

from modules.loan import Loan
from modules.comparator import Comparator

openai.api_key = st.secrets["OPENAI_API_KEY"]

@st.cache_data
def load_loan_data():
    df = pd.read_csv('data/in/tables/loan_data_final.csv')
    return df

def create_pie_chart(labels, values):
    pie = go.Figure(data=[go.Pie(labels=labels, values=values)])
    return pie

@st.cache_data
def process_text_input(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user",
             "content": f'"Extract the loan amount and loan length in months and/or max possible repayment '
                        f'from the following sentence, return only numbers separated by semicolon.'
                        f'"If some of the variables is not in the text, return 0 for it"'
                        f' \n\n {text}'},
        ],
        temperature=0.0
    )

    return response['choices'][0]['message']['content'].split(";")


if 'gpt_ok' not in st.session_state:
    st.session_state.gpt_ok = False

if 'calculated_3' not in st.session_state or st.session_state.calculated_3 is False:
    st.session_state.calculated_3 = False
    st.session_state.available_loans_name_3 = []

if 'loans_ready' not in st.session_state:
    st.session_state.loans_ready = False

if 'loan_amt_gpt' not in st.session_state:
    st.session_state.loan_amt_gpt = 100_000

if 'pay_time_gpt' not in st.session_state:
    st.session_state.pay_time_gpt = 36

if 'pay_amt_gpt' not in st.session_state:
    st.session_state.pay_amt_gpt = None

st.title("Textové vyhledávání")

text_search_input = st.text_input('Zde je možné zadat textový popis:', value='Chci si půjčit 100 000 Kč na 36 měsíců.')

search_button = st.button('Hledat')

if search_button:
    # st.write("Searching for: ", text_search_input)

    if len(text_search_input) < 5:
        st.write("Text je moc krátký.")
        st.stop()

    st.session_state.gpt_output = process_text_input(text_search_input)

    st.session_state.calculated_3 = True

    if len(st.session_state.gpt_output) != 3:
        st.session_state.gpt_ok = False
    else:
        st.session_state.loan_amt_gpt = float(re.findall(r'\d+', st.session_state.gpt_output[0])[0])
        st.session_state.pay_time_gpt = int(re.findall(r'\d+', st.session_state.gpt_output[1])[0])
        st.session_state.pay_amt_gpt = float(re.findall(r'\d+', st.session_state.gpt_output[2])[0])
        if st.session_state.loan_amt_gpt == 0:
            st.session_state.gpt_ok = False
            st.write("V textu nebyla nalezena částka půjčky.")
        elif st.session_state.pay_time_gpt == 0 and st.session_state.pay_amt_gpt == 0:
            st.session_state.gpt_ok = False
            st.write("V textu nebyla nalezena doba splácení ani měsíční splátka.")
        else:
            st.session_state.gpt_ok = True
            if st.session_state.pay_time_gpt > 0:
                st.session_state.pay_amt_gpt = None
            else:
                st.session_state.pay_time_gpt = None

if st.session_state.calculated_3 and st.session_state.gpt_ok is False:
    st.write("Zadali jste všechny údaje? Zkuste to znovu.")
    # st.write("GPT output: ", st.session_state.gpt_output)

if st.session_state.calculated_3 and st.session_state.gpt_ok:
    # st.write("GPT output: ", st.session_state.gpt_output)

    # st.write("GPT output is OK")

    comparator_3 = Comparator(load_loan_data(), st.session_state.loan_amt_gpt, special_type='none',
                              only_banks=False, pay_time=st.session_state.pay_time_gpt)

    st.session_state.available_loans_3 = comparator_3.available_loans

    st.session_state.available_loans_name_3 = comparator_3.available_loans['product_name'].tolist()

    st.session_state.loans_ready = True

    st.session_state.int_rate_3 = float(comparator_3.available_loans['min_rate'][0]/100)

    st.session_state.loan_3 = Loan(st.session_state.loan_amt_gpt, st.session_state.int_rate_3,
                                   loan_length=st.session_state.pay_time_gpt,
                                   max_monthly_payment=st.session_state.pay_amt_gpt)

if st.session_state.loans_ready:
    # Show the best loans
    st.write(f"Nejlepší půjčky pro vás:")

    st.dataframe(data=st.session_state.available_loans_3.style.format(thousands=" ", na_rep="", precision=2),
                 hide_index=True,
                 # use_container_width=True,
                 column_config={
                     "product_name": "Název Produktu",
                     "zk_award": st.column_config.TextColumn(
                         "Ocenění ZK",
                         width="medium"),
                     "min_rate": st.column_config.NumberColumn(
                         "Minimální úrok",
                         format="%.2f %%"
                     ),
                     "delay": "Odklad",
                     "min_amt": st.column_config.NumberColumn(
                        "Minimální částka",
                         format="%.0f Kč"
                     ),
                     "max_amt": st.column_config.NumberColumn(
                        "Maximální částka",
                         format="%.0f Kč"
                     ),
                     "min_len": None,
                     "max_len": None,
                     "non_bank": None,
                     "online": "Sjednání online",
                     "special_cat": None,
                     "link": st.column_config.LinkColumn(
                         "Odkaz"
                     ),
                 })

    selected_loan_3 = st.selectbox('Vyberte půjčku', st.session_state.available_loans_name_3)

    int_rate_3 = float(st.session_state.available_loans_3[
                   st.session_state.available_loans_3['product_name'] == selected_loan_3]['min_rate'].iloc[0])

    st.session_state.loan_3 = Loan(st.session_state.loan_amt_gpt, int_rate_3/100,
                                   loan_length=st.session_state.pay_time_gpt,
                                   max_monthly_payment=st.session_state.pay_amt_gpt)

    st.write(f"Měsíční splátka: {'{:,.2f} Kč'.format(st.session_state.loan_3.monthly_payment).replace(',', ' ')}")
    st.write(f"Celkem úrok: {'{:,.2f} Kč'.format(st.session_state.loan_3.total_interest).replace(',', ' ')}")
    st.write(f"Celková splatná částka:"
             f" {'{:,.2f} Kč'.format(st.session_state.loan_3.total_amount_paid).replace(',', ' ')}")
    st.write(f"Počet splátek: {round(st.session_state.loan_3.payment_plan.Month.max(), 0)}")

    fig = create_pie_chart(['Celková splatná částka', 'Celkem úrok'],
                           [round(st.session_state.loan_3.total_amount_paid, 2),
                            round(st.session_state.loan_3.total_interest, 2)])

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    if st.checkbox('Zobrazit plán splátek'):
        st.subheader("Plán splácení:")

        # Display the output table
        st.dataframe(data=st.session_state.loan_3.payment_plan,
                     hide_index=True,
                     use_container_width=True,
                     column_config={
                        "Month": "Měsíc",
                        "Monthly Payment": st.column_config.NumberColumn("Měsíční splátka", format="%.2f Kč"),
                        "Interest Paid": st.column_config.NumberColumn("Úrok", format="%.2f Kč"),
                        "Principal Paid": st.column_config.NumberColumn("Jistina", format="%.2f Kč"),
                        "Remaining Balance": st.column_config.NumberColumn("Zbývající dluh", format="%.2f Kč")
                     })
