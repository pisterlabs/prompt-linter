# pip install streamlit
# streamlit run 05_13_SQL_generator_web.py
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import text
import openai

st.set_page_config(layout="wide")

st.title('SQL萬用產生器')

col1, col2 = st.columns(2)
with col1:
    prompt = st.text_area('SQL:', value='''
    CREATE TABLE "Orders" 
    (
      "Id" INTEGER PRIMARY KEY, 
      "CustomerId" VARCHAR(8000) NULL, 
      "EmployeesId" INTEGER NOT NULL, 
      "orderdate" VARCHAR(8000) NULL, 
      "RequiredDate" VARCHAR(8000) NULL, 
      "ShippedDate" VARCHAR(8000) NULL, 
      "ShipVia" INTEGER NULL, 
      "Freight" DECIMAL NOT NULL, 
      "ShipName" VARCHAR(8000) NULL, 
      "ShipAddress" VARCHAR(8000) NULL, 
      "ShipCity" VARCHAR(8000) NULL, 
      "ShipRegion" VARCHAR(8000) NULL, 
      "ShipPostalCode" VARCHAR(8000) NULL, 
      "ShipCountry" VARCHAR(8000) NULL 
    );

    CREATE TABLE "Customers" 
    (
      "Id" VARCHAR(8000) PRIMARY KEY, 
      "CompanyName" VARCHAR(8000) NULL, 
      "ContactName" VARCHAR(8000) NULL, 
      "ContactTitle" VARCHAR(8000) NULL, 
      "Address" VARCHAR(8000) NULL, 
      "City" VARCHAR(8000) NULL, 
      "Region" VARCHAR(8000) NULL, 
      "PostalCode" VARCHAR(8000) NULL, 
      "Country" VARCHAR(8000) NULL, 
      "Phone" VARCHAR(8000) NULL, 
      "Fax" VARCHAR(8000) NULL 
    );

    -- find all orders for the customer name = "Alfreds Futterkiste"
    ''', height=300)
    
    button1 = st.button('執行')
    
with col2:
    if button1:
        messages=[
            {"role": "system", "content": "you are a Database expert."},
            {"role": "user", "content": prompt}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0.0,
            messages=messages,
        )
        engine = create_engine("sqlite:///data/Northwind.db", echo=False)
        if '```' in response.choices[0].message.content:
            sql = response.choices[0].message.content.split('```')[1]
        else:
            sql = response.choices[0].message.content.split('\n\n')[1]
            
        sql = sql.replace('\n', ' ')
        st.markdown(f"#### {sql}")
        try:
            df = pd.read_sql(sql, engine)
        except Exception as e:
            st.markdown(f"###### {str(e)}")
        df