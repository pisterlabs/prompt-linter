from main import connect_db
import pandas as pd
from main import get_sql_response
import openai
import os 
import psycopg2
import sys
import config
from datetime import datetime


def connect_db():
    """ Connect to the PostgreSQL database server """
    
    con = None
    try:
        
        # connect to the PostgreSQL server
        
        con = psycopg2.connect(host=config.server_log,
                                database=config.database_log,
                                user=config.user_log,
                                password=config.password_log)	
        # create a cursor
        return (con)
        
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        print ('Could not connect to DB. Exiting..')
        return(error)

def extract_psycopg2_exception(err):
    err_type, err_obj, traceback = sys.exc_info()
    return(err.pgerror, err.pgcode)

def get_data_from_sql(sql_query):
    con = connect_db()
    cur = con.cursor()
    try: 
        cur.execute(sql_query)
    except Exception as err:
        pgerror, pgcode = extract_psycopg2_exception(err)
        print("pgerror: {}".format(pgerror))
        print("pgcode: {}".format(pgcode))


def get_accuracy_stats():
    query ="select  question,sql_generated, count(*) as num_q, \
        sum(case when sql_run_status = 'success' then 1 else 0 end) as num_q_success\
            from sage.sql_log\
                group by question, sql_generated"
    con = connect_db() 
    cur = con.cursor()
    cur.execute(query)
    cnt = cur.fetchall()
    colnames = [desc[0] for desc in cur.description]
    cnt_rec = len(cnt)
    if (cnt_rec>0): 
        df = pd.DataFrame(cnt)
        df.columns = colnames
    else: 
        df = pd.DataFrame(columns = colnames)
    
    cur.close()
    con.close()
    num_q = sum(df.num_q)
    num_q_success = sum(df.num_q_success)

    print("Number of queries {}".format(num_q))
    print("Number of success {}".format(num_q_success))
    return(num_q, num_q_success)

if __name__ == "__main__":
   df = get_accuracy_stats()
   print("Found {} rows".format(df.shape[0]))


