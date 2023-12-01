import streamlit as st
import datetime
from langchain.tools import DuckDuckGoSearchRun
from cg_utils import *

# Get text-to-text FMs
fm_vendors = ['ai21', 'anthropic', 'cohere']
t2t_fms = get_t2t_fms(fm_vendors)


def ask_fm_rag_off(prompt:str, modelid:str):
    """FM query - RAG disabled"""
    if "anthropic.claude" in modelid:
        query = f"\n\nHuman:{prompt}\n\nAnswer:"
    else:
        query = prompt
    fm = get_fm(modelid)
    return fm(query)



def gen_prompt(question:str):
    context = DuckDuckGoSearchRun().run(question)
    PROMPT = f"""
    Human: First use the following context to provide a concise answer to the question at the end. If you cannot find the answer in the context, then provide a concise answer based on your knowledge.

    {context}

    Question: {question}
    Assistant:
    """             
    return PROMPT



def main():
    """Main function for RAG"""
    st.set_page_config(page_title="Retrieval Augmented Generation - Web Search", layout="wide")
    css = '''
        <style>   
            .block-container {
                padding-top: 1rem;
                padding-bottom: 0rem;
                # padding-left: 5rem;
                # padding-right: 5rem;
            }
            button[kind="primary"] {
                background-color: #89AFD7;
                border: none;
            }
            #divshell {
                border-top-right-radius: 7px;
                border-top-left-radius: 7px;
                border-bottom-right-radius: 7px;
                border-bottom-left-radius: 7px;
            }                      
        </style>
    '''
    st.write(css, unsafe_allow_html=True)
    st.header("Retrieval Augmented Generation (RAG) - Web Search")
    st.markdown("Select a foundation model, enter a question about **recent events** to retrieve information from the WWW and press Enter. " \
                 "You will see results with and without using RAG. " \
                 "Refer the [Demo Overview](Solutions%20Overview) for a description of the solution.")
    col1, col2 = st.columns([0.5,2])
    with col1:          
        rag_fm_www = st.selectbox('Select Foundation Model',t2t_fms,key="rag_fm_www_key")
    with col2:
        rag_fm_www_prompt = st.text_input("Enter question or instruction", key="rag_fm_www_prompt_key")
        rag_fm_www_prompt_validation = st.empty()
        st.markdown("<br />", unsafe_allow_html=True)
        prompt = ""
        if rag_fm_www_prompt:
            if len(st.session_state.rag_fm_www_prompt_key) < 10:
                with rag_fm_www_prompt_validation.container():
                    st.error('Your question must contain at least 10 characters.', icon="🚨")
            else:
                prompt = gen_prompt(rag_fm_www_prompt)
        col2_col1, col2_col2 = st.columns([1, 1])
        if prompt:
                with col2_col1:
                        st.markdown(f"<div id='divshell' style='background-color: #fdf1f2;'><p style='text-align: center;font-weight: bold;'>Without RAG ( {rag_fm_www} ) - {datetime.datetime.now().strftime('%d-%b-%Y')}</p>{ask_fm_rag_off(rag_fm_www_prompt, rag_fm_www)}</div>", unsafe_allow_html=True)
                with col2_col2:
                        st.markdown(f"<div id='divshell' style='background-color: #f1fdf1;'><p style='text-align: center;font-weight: bold;'>With RAG ( {rag_fm_www} ) - {datetime.datetime.now().strftime('%d-%b-%Y')}</p>{ask_fm(rag_fm_www, prompt)}</div>", unsafe_allow_html=True)
#
# Main
#    
if __name__ == "__main__":
    main()