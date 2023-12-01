import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import common
import warnings
from openai_adapter import OpenAIAdapter

warnings.simplefilter("ignore")

# ページコンフィグ
st.set_page_config(
    page_title="Generate Text",
    page_icon=":robot_face:",
    layout="wide"
)

# ログインチェック
# common.check_login()
st.warning('ログイン機能を撤廃しました')
st.warning('OpenAI APIは課金制のため，生成は1人1, 2回程度に抑えて頂けると幸いです')
st.info('多用する場合は https://github.com/Daiki04/generate-dock からソースコードをダウンロードして，API Keyを設定してくローカルで実行してください')

# OpenAI Adapter
adapter = OpenAIAdapter()

# モデルの名前
model_names = {"GPT-3.5": "gpt-3.5-turbo-1106", "GPT-4": "gpt-4-1106-preview"}

st.session_state.page_index_n = None
st.session_state.book_title_n = None

# タイトルが空の場合のエラー
if "empty_error" in st.session_state and st.session_state.empty_error == True:
    st.error("タイトルを入力してください")
    st.session_state.empty_error = False

### 生成したい本のタイトルを入力 ###
# labelにはmarkdownを使用
if "submitted" not in st.session_state or st.session_state.submitted == False:
    with st.form("title_form"):
        st.markdown("## 生成したい本のタイトルを入力してください")
        title = st.text_input(":book: 本のタイトル", value="",
                              placeholder="生成したい本のタイトルを入力")
        model_name = st.radio("モデル名：GPT4はGPT3.5と比べると高性能であり，精度の高い回答を行ってくれるバージョン", list(
            model_names.keys()), horizontal=True, index=0)
        temperature = st.slider("温度：温度が低いほど一貫性のある生成が期待でき，高いほど多様性のある生成が期待できる",
                                min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        submitted = st.form_submit_button("生成")

    if submitted:
        if title == "":
            st.session_state.empty_error = True
            st.rerun()
        # st.info(title)
        st.session_state.submitted = True
        st.session_state.title = title
        st.session_state.model_name = model_name
        st.session_state.temperature = temperature
        st.rerun()

### 生成中待機 ###
else:
    if st.session_state.title == "":
        st.session_state.submitted = False
        st.rerun()
    with st.form("title_form"):
        st.markdown("## 生成したい本のタイトルを入力してください")
        title = st.text_input(
            ":book: 本のタイトル", value=st.session_state.title, placeholder="生成したい本のタイトルを入力")
        model_name = st.radio("モデル名：GPT4はGPT3.5と比べると高性能であり，精度の高い回答を行ってくれるバージョン", list(model_names.keys(
        )), index=list(model_names.keys()).index(st.session_state.model_name), horizontal=True)
        temperature = st.slider("温度：温度が低いほど一貫性のある生成が期待でき，高いほど多様性のある生成が期待できる",
                                min_value=0.0, max_value=1.0, value=st.session_state.temperature, step=0.01)
        submitted = st.form_submit_button("生成", disabled=True)
    with st.spinner("""
        生成中，しばらくお待ちください．
        生成が終わると自動的に本棚に移動します．
    """):
        if st.session_state.submitted:
            st.session_state.submitted = False
            adapter.create(
                title, model_names[model_name], temperature=temperature)

    ### 生成完了後，本棚に移動 ###
    st.session_state.page_index_n = 0
    st.session_state.book_title_n = title
    # balloonを表示
    st.session_state.balloon = True
    switch_page("本棚")
