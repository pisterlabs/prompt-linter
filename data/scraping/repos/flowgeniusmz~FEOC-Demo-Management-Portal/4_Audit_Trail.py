import streamlit as st
from  functions.login import get_loginform
from functions.pagesetup import set_title, set_page_overview
import extra_streamlit_components as stx
from functions.audit_data import load_audit_data
from functions.supabase_queries import run_query, supabase_get_audit, supabase_get_notifications
from functions.filter_dataframe import filter_dataframe
import os
import pandas as pd
from langchain.agents import AgentType
from langchain.agents import create_pandas_dataframe_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI



st.set_page_config(layout="wide")

if 'authenticated' not in st.session_state:
    get_loginform()
elif not st.session_state.authenticated:
    get_loginform()
else:
    set_title("FEOC", "Manage Audit Trail")
    set_page_overview("Viewing the Audit Trail", "The audit trail provides the ability for management to view any activity, interaction, log, etc. involved with a FEOC.")
    

    container1=st.container()
    with container1:
        tab_list = [
            stx.TabBarItemData(id=1, title="Audit Log", description="View auditable activity"),
            stx.TabBarItemData(id=2, title="Audit AI", description="Use chat AI for audit"),
            stx.TabBarItemData(id=3, title="Other Audit", description="Miscellaneous audit actions"),
            stx.TabBarItemData(id=4, title="Notifications", description="Notification Log")
        ]
        tab_chosen_id = stx.tab_bar(data=tab_list, default=1)
        
        if tab_chosen_id=="1":
            st.markdown("#### Audit Log CSV")
            dfAudit = load_audit_data()
            dfAuditFilter = filter_dataframe(dfAudit)
            dfAudit_Display = st.dataframe(dfAudit,use_container_width=True, hide_index=False)
        elif tab_chosen_id=="2":
            st.markdown("#### Audit Log Supabase Connection")
            dfAudit2 = supabase_get_audit()
            dfAudit2Filter = filter_dataframe(dfAudit2)
            dfAudit2_Display = st.dataframe(dfAudit2Filter, use_container_width=True, hide_index=False)
            #dfAudit2_Display = st.dataframe(dfAudit2, use_container_width=True, hide_index=False)
        elif tab_chosen_id=="3":
            st.markdown("#### Audit Chat")
            df3 = supabase_get_audit()
            openai_api_key = st.secrets.OPENAI_API_KEY
            if "messages" not in st.session_state:
                st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
            
            for msg in st.session_state.messages:
                st.chat_message(msg["role"]).write(msg["content"])
            
            if prompt := st.chat_input(placeholder="What is this data about?"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.chat_message("user").write(prompt)
                llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", openai_api_key=openai_api_key, streaming=True)
                pandas_df_agent = create_pandas_dataframe_agent(
                    llm,
                    df3,
                    verbose=True,
                    agent_type=AgentType.OPENAI_FUNCTIONS
                )

                with st.chat_message("assistant"):
                    st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                    response = pandas_df_agent.run(st.session_state.messages, callbacks=[st_cb])
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.write(response)
                    
        elif tab_chosen_id=="4":
            st.markdown("#### Notification Log")
            df4 = supabase_get_notifications()
            df4Filter = filter_dataframe(df4)
            df4Display = st.dataframe(df4Filter, use_container_width=True, hide_index=False)

        else:
            st.write("Unknown")
 
    
