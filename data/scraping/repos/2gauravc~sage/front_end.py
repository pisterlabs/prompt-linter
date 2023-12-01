from concurrent.futures.process import _threads_wakeups
import streamlit as st
import os
import openai
from io import StringIO
import pandas as pd
import sys
from datetime import datetime

from main import get_sql_response
from main import get_data_from_sql
from main import get_openai_api_key
from main import log_sql_to_db
from main import get_accuracy_stats
from main import get_failed_sql
from main import update_log_sql_to_db

sage_version = "0.4"

if "row" not in st.session_state:
    st.session_state.row = 0

def next_row():
    st.session_state.row += 1

def prev_row():
    st.session_state.row -= 1

start_tab, main_tab, stat_tab, expert_tab = st.tabs(["Read Me", "Try Sage","Statistics","Expert SQL"])

with start_tab:
    st.header("READ ME")
    st.markdown('###### Just ask a question in English. Sage will \
                translate that into a SQL query and run the query to fetch the result')
    st.markdown('###### Sage demo is set-up with a Sales organization. \
                The organization has customers, products, orders, payments and employees.\
                    The Entity-Relationship diagram of the databse is below.')                

    st.image('images/erdiagram.png', width =600)
    st.markdown('###### Sample Questions to ask:')
    st.markdown('###### 1. Find Customers from France')
    st.markdown('###### 2. Show me unique job titles')
    st.markdown('###### [Download](https://sage-mvp1.s3.amazonaws.com/sales_org_data.xlsx) the data as xl file')
    st.markdown('###### Click on the next tab to try Sage')

with stat_tab:
    @st.cache
    def get_accuracy_stats_fe():
        print("Got Accuracy Metrics get_accuracy_stats_fe")
        return (get_accuracy_stats())

    st.header("ACCURACY STATISTICS")
    st.markdown('#### Sage Accuracy Statistics')
    num_q, num_q_success = get_accuracy_stats_fe()
    success_perc = round(num_q_success / num_q *100,1)
    st.markdown('###### Number of queries generated {}'.format(num_q))
    st.markdown('###### Query Execution Success {}%'.format(success_perc))

    
with main_tab:
    @st.cache
    def fetch_response(query_text: str):
        ## Get the API key  
        openai.api_key = get_openai_api_key()
        ## Get the API response 
        response = get_sql_response(openai.api_key, query_text)
        print("Got SQL Response get_sql_response")
        st.session_state['qgen'] = "SELECT" + response.choices[0]['text'] + '\n'
        #if response is 200 then send the sql  query to the database and get a response 
        
        #Prepare arguments to log_sql_to_db: log_date, question, sql_generated, sage_version, sql_run_status, sql_result_rows
        
        # 
        return (response)

    query_text = ""
    
    st.session_state.dis_fetch = True

    st.markdown('## Sage Version  **V0.1**')

    st.markdown("#### Ask a question")

    qu = st.text_area(label="Ask in natural language", key='qask')
    #print (qu) 

    if len(qu) > 0:
        st.session_state.dis_fetch = False

    res = st.button("Fetch", on_click=fetch_response, args=(qu,), disabled=st.session_state['dis_fetch'])

    if 'qgen' in st.session_state: 
        st.markdown("#### Query Generated")
        qdis = st.text_area(label='Query Returned by Sage',value = st.session_state['qgen'],key='qdis', height = 100)
        
        status,df, err_dict = get_data_from_sql(st.session_state['qgen'])
        print("Got fata from sql get_data_from_sql")
        if status == 'success':
            row_cnt_txt = 'Found {} rows'.format(df.shape[0])
            st.session_state['row_cnt_txt'] = row_cnt_txt
            st.session_state['ret_df'] = df
            st.session_state['sql_run_status'] = "success"
            st.session_state['sql_result_rows'] =  df.shape[0]
        
        else: #status is error
            st.session_state['sql_run_status'] = "failure"
            st.session_state['sql_result_rows'] = 0
            
        st.session_state['pgerror'] = err_dict['pgerror']
        st.session_state['pgcode'] = err_dict['pgcode']
    

    if 'sql_run_status' in st.session_state: 
        st.markdown("#### Data Result")
        if st.session_state['sql_run_status'] == "failure":
            st.markdown("###### Query failed to execute. The error code is {}".format(st.session_state['pgcode']))
        elif (st.session_state['sql_run_status'] == "success") & ('ret_df' in st.session_state):
            st.markdown("###### {}".format(st.session_state['row_cnt_txt']))
            st.dataframe(st.session_state['ret_df'])
        else:
             st.markdown("###### Query executed, but no data returned. The error has been logged")

        
    if 'qgen' in st.session_state:
        try: 
            log_date = datetime.today().strftime('%Y-%m-%d')
            question = qu
            sql_generated = st.session_state['qgen']
            sage_version = "0.3"
            log_sql_to_db(log_date, question, \
                sql_generated, sage_version, st.session_state['sql_run_status'], st.session_state['sql_result_rows'], \
                    st.session_state['pgerror'], st.session_state['pgcode'])
            print ("Logged to DB log_sql_to_db")
        except (Exception) as error:
            print ("Error writing to log DB")
            print(error)
        del st.session_state['qgen']

with expert_tab:
    @st.cache
    def get_failed_sql_fe():
        print("Getting failed SQL get_failed_sql_fe")
        return(get_failed_sql())
    
    @st.cache
    def try_save_expert_sql(log_id, esql):
        status,df, err_dict = get_data_from_sql(esql)
        if status == 'success':
            row_cnt_txt = 'Found {} rows'.format(df.shape[0])
            print ("Expert SQL success. {}".format(row_cnt_txt))
            update_log_sql_to_db(log_id, esql)
        else: #status is error
            print ("Expert SQL failed. error code is {}".format(err_dict['pgcode']))
            
        return

    st.header("Expert SQL")
    st.markdown('###### Help train Sage. Give us the right SQL.')
    df = get_failed_sql_fe()
    st.markdown('###### Found {} failed SQLs'.format(df.shape[0]))
    test = df.astype(str)
    col1, col2, _, _ = st.columns([0.1, 0.17, 0.1, 0.63])
    
    if "row" not in st.session_state:
        st.session_state.row = 0
    
    if st.session_state.row < len(df.index)-1:
        col2.button(">", on_click=next_row)
    else:
        col2.write("") 
    
    if st.session_state.row > 0:
        col1.button("<", on_click=prev_row)
    else:
        col1.write("") 
    #print(st.session_state.row)

    st.write(test.iloc[st.session_state.row])

    st.session_state.dis_esql_try = True

    exp_sql = st.text_area(label="Provide the Correct SQL", key='esql')
    #print (qu) 

    if len(exp_sql) > 0:
        st.session_state.dis_esql_try = False

    res = st.button("Try this instead", on_click=try_save_expert_sql, args=(test.iloc[st.session_state.row,0],exp_sql), disabled=st.session_state['dis_esql_try'])
