import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from google.cloud import translate
from google.oauth2.service_account import Credentials
import pinecone
import openai

st.set_page_config(
    page_title="home",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.markdown(
    """
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
</style>
""",
    unsafe_allow_html=True,
)
st.markdown(
    """
    <style>
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
    .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
    .viewerBadge_text__1JaDK {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource(show_spinner=None)
def init_openai_key():
    openai.api_key = st.secrets.OPENAI_TOKEN

    return openai.api_key

def init_gcp_connection():
    credentials = Credentials.from_service_account_info(st.secrets.GOOGLE)
    translate_client = translate.TranslationServiceClient(credentials=credentials)

    return translate_client

def init_pinecone_connection():
    pinecone.init(
        api_key=st.secrets["PINECONE_KEY"],
        environment=st.secrets["PINECONE_REGION"]
    )
    pinecone_index = pinecone.Index('bookstore')
    return pinecone_index


@st.cache_resource(show_spinner=None)
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


if __name__ == '__main__':
    openai.api_key = init_openai_key()
    pinecone_client = init_pinecone_connection()

    st.subheader(":house:", anchor="home")

    empty0, con0, empty7 = st.columns([0.3, 0.5, 0.3])
    with empty0:
        st.empty()
    with con0:
        st.image("./home_img.png")
    with empty7:
        st.empty()

    container = st.container()
    with container:
        empty1, con1, empty2 = st.columns([0.3, 1.0, 0.3])
        with empty1:
            st.empty()
        with con1:
            e1, e2, e3 = st.columns(3)
            with e2:
                local_css("./button_style.css")
                want_to_contribute = st.button("플레이리스트 만들기 >")
                if want_to_contribute:
                    switch_page("main")
        with empty2:
            st.empty()
