import openai
import streamlit as st
from langchain import OpenAI
from langchain.agents import create_sql_agent, AgentType
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header

import config
import helper

st.set_page_config(page_title='人人都是数据分析师', layout='wide', page_icon='🤖')

# 侧边栏配置
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API key：[有默认值]", key="set_api_key", placeholder="点击输入")
    st.selectbox("大语言模型：", index=0, options=config.MODEL_OPTIONS, key="select_model")
    # 加载数据源
    st.write("\n")
    st.markdown("### 🕋 选择数据源")
    st.selectbox("数据源加载：", index=0, options=config.DATA_SOURCES, key="select_data_source")
    if st.session_state['select_data_source'] == '本地文件[CSV]':
        data_lst, metadata_lst = helper.load_offline_file()
        st.session_state['data_source'] = 'offline'
    elif st.session_state['select_data_source'] == 'MySQL':
        data_lst = False
        st.session_state['data_source'] = 'mysql'
        # 聊天对话表单
        with st.form("sql_chat_input", clear_on_submit=True):
            user = st.text_input(
                label="用户名",
                placeholder="输入用户名：",
                label_visibility="collapsed",
                key='user_name'
            )
            password = st.text_input(
                label="用户密码",
                placeholder="输入密码：",
                label_visibility="collapsed",
                key='user_password'
            )
            host = st.text_input(
                label="主机IP",
                placeholder="输入主机IP：",
                label_visibility="collapsed",
                key='host_ip'
            )
            port = st.text_input(
                label="端口号",
                placeholder="输入端口号：",
                label_visibility="collapsed",
                key='port'
            )
            db_name = st.text_input(
                label="数据库名称",
                placeholder="输入数据库名称：",
                label_visibility="collapsed",
                key='db_name'
            )
            submitted = st.form_submit_button("提交", use_container_width=True)
        if submitted:
            data_lst = True
            sql_uri = 'mysql+pymysql://{user}{password}@{host}:{port}/{db_name}'.format(user=user,
                                                                                        password=':' + password,
                                                                                        host=host, port=port,
                                                                                        db_name=db_name)
            db = SQLDatabase.from_uri(sql_uri)
            toolkit = SQLDatabaseToolkit(db=db, llm=OpenAI(temperature=0))
            agent_executor = create_sql_agent(
                llm=OpenAI(temperature=0),
                toolkit=toolkit,
                verbose=True,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            )
            st.session_state['agent_executor'] = agent_executor
    else:
        assert False, "数据源加载失败！"
    st.write("---")
    st.markdown('<a href="https://github.com/typole/AB-AutoGPT" target="_blank" rel="ChatGPT-Assistant">'
                '<img src="https://badgen.net/badge/icon/GitHub?icon=github&amp;label=AB-AutoGPT" alt="GitHub">'
                '</a>', unsafe_allow_html=True)

# 主页面内容
st.subheader("💹 人人都是数据分析师")
st.caption("实战全流程：业务指标 → 指标量化 → 数据探索 → 数据建模 → 数据可视化 → 观点输出 → 业务指标 → ...")

tap_chat, tap_example, tap_meta, tap_chart, tap_methodology = st.tabs(
    ['👆 数据探索', '👉 数据示例', '👇 元数据', '👉 数据可视化', '👊 分析方法论'])
with tap_chat:
    if not data_lst:
        st.caption("请配置数据源，并加载数据！")
    else:
        st.write("数据源已加载！开始你的数据探索之旅吧！")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "有什么我能帮助您？"}]

    with st.form("csv_chat_input", clear_on_submit=True):
        a, b = st.columns([4, 1])
        user_input = a.text_input(
            label="请输入:",
            placeholder="你想和我聊什么?",
            label_visibility="collapsed",
        )
        b.form_submit_button("Send", use_container_width=True)

    for msg in st.session_state.messages:
        colored_header(label='', description='', color_name='blue-30')
        message(msg["content"], is_user=msg["role"] == "user")

    if openai_api_key:
        openai.api_key = openai_api_key
    else:
        openai.api_key = st.secrets['OPENAI_API_KEY']

    if user_input and data_lst != []:
        st.session_state.messages.append({"role": "user", "content": user_input})
        message(user_input, is_user=True)
        if st.session_state['data_source'] == 'offline':
            agent = helper.built_agent_llm(data_lst)
        else:
            agent = st.session_state['agent_executor']
        try:
            response = agent.run(user_input)
        except Exception as e:
            assert e
        else:
            st.session_state.messages.append(response)
            message(response)
    else:
        st.caption("请配置数据源，并加载数据！")

# with tap_example:
#     if data_lst:
#         option = st.selectbox("选择数据对象：", index=0, options=metadata_lst, key="select_metadata_example")
#         for idx in range(len(metadata_lst)):
#             if metadata_lst[idx] == option:
#                 st.data_editor(data_lst[idx], height=600)
#     else:
#         st.caption("请配置数据源，并加载数据！")
#
# with tap_meta:
#     if data_lst:
#         option = st.selectbox("选择数据对象：", index=0, options=metadata_lst, key="select_metadata_meta")
#         for idx in range(len(metadata_lst)):
#             if metadata_lst[idx] == option:
#                 st.markdown("#### 数据统计")
#                 st.data_editor(data_lst[idx].describe(), height=600)
#     else:
#         st.caption("请配置数据源，并加载数据！")

with tap_chart:
    if not data_lst:
        st.caption("请配置数据源，并加载数据！")
    else:
        st.write("敬请期待！")

with tap_methodology:
    st.caption("敬请期待！")
