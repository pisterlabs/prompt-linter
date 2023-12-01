import streamlit as st
from langchain import PromptTemplate
from langchain.llms import OpenAI
import os

import hmac

openai_api_key = st.secrets["openai_api_key"]



# streamlit_app.py



def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the passward is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


if not check_password():
    st.stop()  # Do not continue if check_password is not True.

# Main Streamlit app starts here


template = """

Sua tarefa é ser um tradutor entre duas variantes de português:  português europeu e português brasileiro. Vou enviar textos em uma das versões, assim como a versão em que desejo traduzir.

Seu objetivo é:
- Identificar a versão de português usada no texto original
- Re-escrever o texto na versão desejada do português, fazendo uma tradução mas mantendo seu significado original.

Seguem alguns exemplos de palavras e expressões em versões de português que tem o mesmo significado:
- Português europeu: autocarro, t1, telemóvel, esquentador, gelado, frigorífico, chavala, por agora, a fazer, amo-te
- Português brasileiro: ônibus, apartamento de 1 quarto, celular, aquecedor, sorvete, geladeira, garota, por enquanto, fazendo, te amo

Texto de input:
{input}

Versão desejada: 
{target}

Versão traduzida:
"""

prompt = PromptTemplate (
    input_variables = ["input", "target"],
    template=template,
)

st.title("Tradutor Zuca 🇧🇷🇵🇹 Tuga")


def load_LLM():
    llm = OpenAI(temperature=0.7, openai_api_key= openai_api_key)
    return llm
    
llm = load_LLM()

def get_text():
    input_text = st.text_area("Texto para tradução:", "Escreva ou cole o texto que deseja traduzir")
    return input_text

def get_original():
    original = st.selectbox("Como quer traduzir?", ("Português brasileiro ⇾ europeu", "Português europeu ⇾ brasileiro"))
    return original

def get_target(original):   
    if original == "Português europeu":
        target = "Português brasileiro"
    else:
        target = "Português europeu"
    return target

with st.form("input-form"):

    original = get_original()
    input = get_text()
    submitted = st.form_submit_button(label="Traduzir")
   
    if submitted:
        target = get_target(original)
        prompt_with_text = prompt.format(input=input, target=target)
        converted = llm(prompt_with_text)
        #converted = "Teste"
        st.write (f"**Texto original:**")
        st.write (input)
        st.write (f"**Texto traduzido:**")
        st.write (converted)
       
        
        
        