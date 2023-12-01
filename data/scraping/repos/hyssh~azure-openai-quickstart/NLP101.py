import os
import openai
import dotenv
import pandas as pd
import streamlit as st
from sqldbconn import get_connx

# .env file must have OPENAI_API_KEY and OPENAI_API_BASE
dotenv.load_dotenv()
openai.api_type = "azure"
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_version = "2023-03-15-preview"
openai.api_key = os.getenv("OPENAI_API_KEY")
ENGINE = os.getenv("ENGINE")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")
MODEL_NAME = os.getenv("MODEL_NAME")

def sample101():
    system_char = """
    You are a data analyst working for a company that provides data analysis services to clients. 
    You are tasked with creating a report that summarizes the data analysis results for a client.    
    """

    database_summary = """
    ## Database Summary
    Database has ten tables
    SalesLT.Address(AddressID, AddressLIne1, AddressLine2, City, StateProvince, CountryRegion, PostalCode, rowguid, modifiedDate)
    SalesLT.Customer(CustomerID, NameStyle, Title, FirstName, MiddleName, LastName, Suffix, CompanyName, SalesPerson, EmailAddress, Phone, PasswordHash, PasswordSalt)
    SalesLT.CustomerAddress(CustomerID, AddressID, AddressType)
    SalesLT.Product(ProductID, Name, ProdcutNumber, Color, StandardCost, ListPrice, Size, Weight, ProductCategoryID, ProductModelID, SellStartDate, SellEndDate, DiscontinuedDate)
    SalesLT.ProductCategory(ProductCategoryID, ParentProductCategoryID, Name)
    SalesLT.ProdcutDescription(ProductDescriptionID, Description)
    SalesLT.ProductModel(ProductModelID, Name, CatalogDesciortion)
    SalesLT.ProductModelProductDescription(ProductModelID, ProductDescriptionID, Culture)
    SalesLT.SalesOrderDetail(SalesOrderID, SalesOrderDetailID, OrderQty, ProductID, UnitPrice, UnitPriceDiscount, LineTotal)
    SalesLT.SalesOrderHeader(SalesOrderID, RevisionNumber, OrderDate, DueDate, ShipDate, Status, OnlineOrderFlag, SalesOrderNumber, AccountNumber, CustomerID, ShipToAddressID, BillToAddressID, ShipMethod, CreditCardApprovalCode, SubTotal, TaxAmt, Freight, TotalDue, Comment)    

    ## Safty Instruction
    Do not run any DML or DDL queries. Help user to get the information they need.
    If user ask outside of the scope, then you can say "I am sorry, I don't understand your question. Can you please rephrase your question?"
    Do not show customer's EmailAddress, Phone, PasswordHash, PasswordSalt, but is okay to share their our product purchase history or amounts.
    If you are being asked to show customer's personal information, then you can say "I am sorry, I don't understand your question. Can you please rephrase your question?"

    Write a T-SQL query to get the following information:
    """

    # concat two strings into one
    system_instruction = system_char + database_summary

    # st.set_page_config(page_title="Natual Language to SQL query", page_icon=":robot_face:", layout="wide", initial_sidebar_state="collapsed")

    st.markdown("# Natural Language to SQL Query")
    st.markdown("This demo will show you how to use Azure OpenAI to convert natural language to SQL query")
    with st.expander("Demo scenario"):
        st.image("https://github.com/hyssh/azure-openai-quickstart/blob/main/images/Architecture-demo.png?raw=true")
        st.markdown("1. User will type a question in the input box")
        st.markdown("2. __Web App__ sends the question to __Azure OpenAI__")
        st.markdown("3. __Azure OpenAI__ will convert the question to SQL query")
        st.markdown("4. __Web App__ sends the SQL query to __Azure SQL DB__")
        st.markdown("5. __Azure SQL DB__ will execute the SQL query and return the result to __Web App__")
        st.markdown("6. __Web App__ will show the result to user")

    st.markdown("---")
    with st.container():
        # create a input using streamlit
        inputmsg = st.text_input("Type your question")

    with st.sidebar:
        with st.container():
            st.info("Simple natural language to SQL query converter")
        sample_tab, prompt_tab, system_tab = st.tabs(["samples", "prompts", "system"])
        # samples
        with sample_tab:
            st.write("Sample Questions ")
            st.code("Get 10 customer who purchased our products the most with the amount of the purchases", language="html")
            st.code("Get customer email and address who purchased our products the most with the amount of the purchases", language="html")
            st.code("Create a new table for customer support", language="html")
            st.code("Add 100 to all unit price of our products", language="html")
            st.code("Drop Customer table", language="html")

        # prompts
        with prompt_tab:
            st.header("Prompts")
            st.code(database_summary, language="html")

        with system_tab:
            st.header("System messages")
            st.code(system_char, language="html")

    # if input is not empty, then run the code below
    if inputmsg:
        with st.spinner("Loading..."):
            system_msgs = {"role":"system","content":system_instruction}
            user_msg = {"role":"user","content":inputmsg}
            prompt = [system_msgs, user_msg]

            response = openai.ChatCompletion.create(
                engine=ENGINE,
                messages = prompt,
                temperature=0.1,
                max_tokens=400,
                top_p=0.00,
                n=3,
                frequency_penalty=1.0,
                presence_penalty=1.0,
                stop=None)

            # extract sql query part from the response open ai
            try:
                sql_query = response.choices[0].message['content'].split("```")[1]
                with st.expander("See executed Query"):
                    st.write(sql_query)
                df = pd.read_sql_query(sql_query, get_connx())
                st.dataframe(df)
            except:
                st.write(response.choices[0].message['content'])
            
            with st.expander("See SQL Query explanation"):
                st.write(response.choices[0].message['content'])

if __name__ == "__main__":
    sample101()