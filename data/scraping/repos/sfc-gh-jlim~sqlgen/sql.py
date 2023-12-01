
import openai
import streamlit as st
import os
# import requests
from dotenv import load_dotenv

def gpt3(sql_prompt):
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt="generate a sql statement to" + sql_prompt,
        max_tokens=256,
        temperature=0.6,
        )
    answer = response.choices[0]['text']
    return answer

def main():
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_API_KEY')

    st.header("Never Write SQL Again")

    st.header("This is an amazing SQL Generator, powered by GPT-3")

    st.write("""
    Here are some examples:
    - Input: count the number of staff under managers and sort from most to least
    - Output: 
    """)
    st.code("""
    SELECT COUNT(*) AS 'Number of Staff', manager_id 
    FROM staff 
    GROUP BY manager_id 
    ORDER BY 'Number of Staff' DESC    
    """)

    st.write("""
    - Input: calculate who are my top customers by region that requires joining sales and customers table
    - Output: 
    """)


    st.code("""SELECT c.customer_name, c.region, SUM(s.total_sales) AS total_sales 
FROM customers c 
JOIN sales s ON c.customer_id = s.customer_id 
GROUP BY c.customer_name, c.region 
ORDER BY total_sales DESC;""")

    with st.form("my_form"):

        sql_request = st.text_area("What SQL statement do you like to generate ?",value="count the number of branches under each regions and sort their sales in a descending order")

        submitted = st.form_submit_button("Submit")
        
        if submitted:
            st.header("Your SQL")
            sql_response = gpt3(sql_request)
            st.write(sql_response)


if __name__ == "__main__":
    main()