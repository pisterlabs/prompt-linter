#Author: Ravish Garg
#Customer Engineer, Data Specialist

'''
Find the car along with km_driven of year 2018 with the lowest km driven and highest mileage

Find the name of the top 5 car after 2018 and has less than 50000 km driven

Generate a table of the count of cars based on fuel type after 2015

Generate a Bar graph of cars with fuel as column and count of cars as data values

Generate a bar chart of cars with year as column and count of cars as data values
'''

import streamlit as st
import streamlit.components.v1 as stc
import pandas as pd
import secrets
import numpy as np
import pandas.io.sql as sqlio
import pandasql as ps
import sys
import os
import locale
from datetime import datetime
import psycopg2
import math
import json
import ast
import time
from PIL import Image
from langchain.llms import GooglePalm
from langchain.llms import VertexAI
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.agents import AgentExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import VertexAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document
from google.cloud import aiplatform
import vertexai
from vertexai.language_models import TextGenerationModel,TextEmbeddingModel
from google.cloud.alloydb.connector import Connector
import asyncio
import asyncpg
from pgvector.asyncpg import register_vector
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from psycopg2.extensions import register_adapter, AsIs, adapt, new_type, register_type
from PIL import Image

if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO

st.set_page_config(page_title='GCP DEMO - Cars 4 Sale', page_icon=':smiley:')
locale.setlocale(locale.LC_ALL, '')

page_bg_img = """
<style>
.reportview-container {
background-image: linear-gradient(rgba(0, 0, 0, 0.7),
                       rgba(0, 0, 0, 0.7)),url("https://images.unsplash.com/photo-1621831955776-6ce162d24933?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1974&q=80");
background-size: cover;
}
.sidebar .sidebar-content {
   display: flex;
   align-items: center;
   justify-content: center;
}
</style>
"""
#st.markdown(page_bg_img, unsafe_allow_html=True
def addapt_numpy_float64(numpy_float64):
    return AsIs(numpy_float64)
def addapt_numpy_int64(numpy_int64):
    return AsIs(numpy_int64)
def addapt_numpy_array(numpy_array):
    return AsIs(tuple(numpy_array))
register_adapter(np.float64, addapt_numpy_float64)
register_adapter(np.int64, addapt_numpy_int64)
register_adapter(np.ndarray, addapt_numpy_array)


project_id = secrets.project_id  # @param {type:"string"}
region = secrets.region
endpoint = secrets.endpoint

hostname = secrets.hostname
port_num = secrets.port_num
db_name = secrets.db_name
alloydb_user = secrets.alloydb_user
alloydb_user_pswd = secrets.alloydb_user_pswd

chk = 0

def inventory():
    st.write("#### Cars at best prices...")
    conn = psycopg2.connect(host=hostname, port=port_num, dbname=db_name, user=alloydb_user, password=alloydb_user_pswd)
    sql = "select * from car_details;"
    dat = sqlio.read_sql_query(sql, conn)
    conn = None
    #st.dataframe(dat)
    #<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    card_template = """
                    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
                        <div class="card bg-light mb-3" >
                            <H5 class="card-header">{}.  <a href={} style="display: inline-block" target="_blank">{}</h5>
                                <div class="card-body">
                                    <span class="card-text"><b>Year: </b>{}</span><br/>
                                    <span class="card-text"><b>Selling_Price (INR): </b>{}</span><br/>
                                    <span class="card-text"><b>Owner: </b>{}</span><br/>
                                    <span class="card-text"><b>Seller_Type: </b>{}</span><br/>
                                    <span class="card-text"><b>Mileage (kmpl): </b>{}</span><br/>
                                    <span class="card-text"><b>Transmission: </b>{}</span><br/><br/>
                                    <p class="card-text"><b>No. of seats: </b>{}
                                    <b>,Max_power (bhp): </b>{}
                                    <b>,KM Driven: </b>{}
                                    <b>,Fuel: </b>{}
                                    <b>,Estimated Price: </b>{}
                                    </p>
                                </div>
                            </div>
                        </div>
            """
    #paper_url='https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css'
    paper_url=''
    for index, row in dat.iloc[8120:].iterrows():
        st.markdown(card_template.format(str(index + 1), paper_url, row['name'], row['year'], row['selling_price'], row['owner'], row['seller_type'], row['mileage'], row['transmission'], row['seats'], row['max_power'], row['km_driven'], row['fuel'], row['predicted_selling_price']), unsafe_allow_html=True)        
    #conn.close()

def gen_embeddings():
    '''
    conn = psycopg2.connect(host=hostname, port=port_num, dbname=db_name, user=alloydb_user, password=alloydb_user_pswd)
    sql = "select * from car_details;"
    dat = sqlio.read_sql_query(sql, conn)
    conn = None
    print(type(dat))
    st.write(dat.head())
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[".", "\n"],
        chunk_size=500,
        chunk_overlap=0,
        length_function=len,
    )
    chunked = []
    dat2 = pd.DataFrame(dat['id'])
    dat = dat.drop(['id'],axis=1)
    dat2['Desc'] = dat[dat.columns[1:]].apply(lambda x: ','.join(x.dropna().astype(str)),axis=1)
    st.write(dat2.head())
    for index, row in dat2.iterrows():
        product_id = row["id"]
        model_name = row["Desc"]
        splits = text_splitter.create_documents([model_name])
        for s in splits:
            r = {"id": product_id, "content": s.page_content}
            chunked.append(r)

    model = TextEmbeddingModel.from_pretrained("textembedding-gecko")

    batch_size = 5
    for i in range(0, len(chunked), batch_size):
        request = [x["content"] for x in chunked[i : i + batch_size]]
        response = model.get_embeddings(request)
        for x, e in zip(chunked[i : i + batch_size], [embedding.values for embedding in response]):
            x["embedding"] = e
    
    print(chunked)
    product_embeddings = pd.DataFrame(chunked)
    print(product_embeddings.head())
    print ("\n Number of vector rows generated: ",product_embeddings.shape[0])
    conn2 = psycopg2.connect(host=hostname, port=port_num, dbname=db_name, user=alloydb_user, password=alloydb_user_pswd)
    with conn2:
        register_vector(conn2)
        cur = conn2.cursor()
        data_list = [(int(row['id']), row['content'], np.array(row['embedding'])) for index,row in product_embeddings.iterrows()]
        execute_values(cur, "INSERT INTO embeddings (id, content, embedding) VALUES %s", data_list)
        conn2.commit()
        print('Embedding has been uploaded...')
    '''
    image = Image.open('Alloydb_Vector_Langchain.png')
    st.image(image,width=1100)
    conn2 = psycopg2.connect(host=hostname, port=port_num, dbname=db_name, user=alloydb_user, password=alloydb_user_pswd)
    with conn2:
        cur = conn2.cursor()
        query1 = "select count(1) from car_details"
        query2 = "select * from embeddings limit 2"
        cur.execute(query1)
        op1 = cur.fetchall()
        st.write("### Number of records in the database: ",op1[0])
        rec = int((str(op1[0])).replace(",","").replace("(","").replace(")",""))
        print(rec)
        print(type(rec))
        numdim = rec * 768
        st.write("### Number of dimensions generated based on given records: ",f'{numdim:n}')
        cur.execute(query2)
        op2 = cur.fetchall()
        st.write("### First two generated embeddings:")
        st.write(op2)
        cur.close()
    conn2.close()

def search_embedding():
    st.write("## Embedding based analytics:")
    analytic_query = str(st.text_input("Enter your query: "))
    llm_type = str(st.selectbox('Grounding:',('Yes','No')))
    if st.button("Submit"):
        if llm_type == 'Yes':
            embeddings_service = VertexAIEmbeddings()
            qe = np.array(embeddings_service.embed_query([analytic_query]))
            print('\n')
            print(qe)
            print('\n')
            conn3 = psycopg2.connect(host=hostname, port=port_num, dbname=db_name, user=alloydb_user, password=alloydb_user_pswd)
            similarity_threshold = 0.65
            num_matches = 5
            with conn3:
                register_vector(conn3)
                cur = conn3.cursor()
                cur.execute(
                    """
                        WITH vector_matches AS (
                            SELECT id, 1 - (embedding <=> %(vector)s) AS similarity
                            FROM embeddings
                            WHERE 1 - (embedding <=> %(vector)s) > %(thres)s
                            ORDER BY similarity DESC
                            LIMIT %(nm)s
                        )
                        SELECT cd.name,cd.year,cd.km_driven,cd.fuel,cd.seller_type,cd.transmission,cd.owner,cd.mileage,cd.engine,cd.max_power,cd.torque,cd.seats,emb.similarity FROM car_details cd,vector_matches emb WHERE cd.id IN (SELECT id FROM vector_matches);
                    """,{'vector':qe,'thres':similarity_threshold,'nm':num_matches}
                )
                results = cur.fetchall()
                st.write(results)
                cur.close()
                print('Done...')        
            conn3.close()
        elif llm_type == "No":
            parameters = {
                "temperature": 0.2,
                "max_output_tokens": 512,
                "top_p": 0.8,
                "top_k": 40
            }
            model = TextGenerationModel.from_pretrained("text-bison@001")
            response = model.predict(
            analytic_query,
            **parameters,
            )
            st.write(response.text)
            print('Done!')

def callback(engine,max_power,torque,seats):
    st.session_state.engine = engine
    st.session_state.max_power = max_power
    st.session_state.torque = torque
    st.session_state.seats = seats

def sell_car():
    if 'engine' not in st.session_state:
        st.session_state['engine']=0
    if 'max_power' not in st.session_state:
        st.session_state['max_power']=0
    if 'torque' not in st.session_state:
        st.session_state['torque']=0
    if 'seats' not in st.session_state:
        st.session_state['seats']=0

    engine = st.session_state.engine
    max_power = st.session_state.max_power
    torque = st.session_state.torque
    seats = st.session_state.seats

    print(engine,type(engine))
    print(max_power,type(max_power))
    print(torque,type(torque))
    print(seats,type(seats))

    st.write("## Provide your car details:")
    name = str(st.text_input("Car Model Name: "))
    year = int(st.slider("Year: ",1990,2022,2015))
    km_driven = int(st.number_input("KM Driven: "))
    fuel = str(st.selectbox('Fuel:',('Diesel','Petrol','CNG','EV','LPG')))
    seller_type = str(st.selectbox('Seller_Type: ',('Dealer','Individual','TrustmarkDealer')))
    transmission = str(st.selectbox('Transmission: ',('Manual','Automatic')))
    owner = str(st.selectbox('Owner: ',('FirstOwner','SecondOwner','ThirdOwner','Fourth&AboveOwner','TestDriveCar')))
    mileage = int(st.number_input("Mileage (kmpl): "))
    if st.button("Help Me!"):
        prompt =  "I want the following specs data for a specific car as per given format:"
        if(engine >= 0):
            prompt += "\n Engine(cc) in numbers only with key as 'Engine'."               
        if(max_power >= 0):
            prompt += "\n Max_Power(bhp) in numbers only with key as 'Max_Power'."    
        if(torque == '0'):
            prompt += "\n Torque in numbers only with key as 'Torque'."  
        if(seats >= 2):
            prompt += "\n Number of seats in numbers only." 
        print(prompt,name,year)
        prompt += "\n Given the following information:\n Model:" + str(name) + "\n Year of manufacturing" + str(year) + "\n Give me the above mentioned specs for the given vehicle in json format, stick to only the list that I provided earlier and if any of Engine,Max_power,Torque, Number of seats not available set it as zero."
        parameters = {
            "temperature": 0.2,
            "max_output_tokens": 512,
            "top_p": 0.8,
            "top_k": 40
        }
        model = TextGenerationModel.from_pretrained("text-bison@001")
        response = model.predict(
        prompt
        ,
        **parameters,
        )
        output=response.text
        print(output)
        output = output.replace("```","")
        output = ast.literal_eval(output)
        print(type(output))
        if 'Engine' in output.keys():
            engine = output['Engine']
        if 'Max_Power' in output.keys():
            max_power = output['Max_Power']
        if 'Torque' in output.keys():
            torque = output['Torque']
        if 'Number_of_Seats' in output.keys():
            seats = output['Number_of_Seats']
        st.write("\n\n")
        st.write("### Here is the vehicle missing info...")
        st.write(output)
        st.write("#### Proceed with predict...")
    engine = int(st.number_input("Engine (cc): ",value=int(engine)))
    max_power = int(st.number_input("Max_Power (BHP): ",value=int(max_power)))
    torque = str(st.text_input("Torque (Optional): ",value=int(torque)))
    idx=2
    if seats == 2:
        idx=0
    elif seats == 4:
        idx=1
    elif seats ==5:
        idx=2
    elif seats ==6:
        idx=3
    elif seats ==7:
        idx=4
    elif seats ==8:
        idx=5
    seats = int(st.selectbox('No. of Seats: ',(2,4,5,6,7,8),index=idx))


    if st.button("Predict",on_click=callback(engine,max_power,torque,seats)):
        st.write("### Uploading the given data points...")
        if torque is None:
            torque = "NA"
        
        conn2 = psycopg2.connect(host=hostname, port=port_num, dbname=db_name, user=alloydb_user, password=alloydb_user_pswd)
        with conn2:
            cur = conn2.cursor()
            #st.write(name,year,km_driven,fuel,seller_type,transmission,owner,mileage,engine,max_power,torque,seats)
            print(name,year,km_driven,fuel,seller_type,transmission,owner,mileage,engine,max_power,torque,seats)
            cur.execute("insert into car_details(name,year,km_driven,fuel,seller_type,transmission,owner,mileage,engine,max_power,torque,seats) values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",(name,year,km_driven,fuel,seller_type,transmission,owner,mileage,engine,max_power,torque,seats))
            st.write("#### Computing the best price...")
            q2=("select predicted_selling_price::json->'predictions'->0->'value' as predicted_selling_price from (select ml_predict_row('projects/%(project_id)s/locations/%(region)s/endpoints/%(endpoint)s', json_build_object('instances',json_build_array(json_build_object('name',cd.name,'year',cast(cd.year as VARCHAR),'km_driven',cast(cd.km_driven as VARCHAR),'fuel',cd.fuel,'seller_type',cd.seller_type,'transmission',cd.transmission,'owner',cd.owner,'mileage',cd.mileage,'engine',CAST(cd.engine AS VARCHAR),'max_power',cd.max_power,'torque',cd.torque,'seats',CAST(cd.seats AS VARCHAR))))) as predicted_selling_price from car_details cd where id = (select max(id) from car_details)) as json_query",{'project_id':project_id,'region':region,'endpoint':endpoint})
            cur.execute(q2)
            psp = cur.fetchone()
            st.write("### Best possible price (INR): ",psp[0])
            predicted_price = psp[0]
            predicted_price = math.floor(predicted_price)
            print(f"Predicted Price: {predicted_price}")
            q3="update car_details set predicted_selling_price = %s where id = (select max(id) from car_details)"
            cur.execute(q3,[predicted_price])
            print(f"Number of rows updated: {cur.rowcount}")
            st.write("Done.")
        cur.close()
        conn2.close()

    else:
        st.write("Get the best price for your car...")

def analytics():
    st.write("## Analytics Platform:")
    db = SQLDatabase.from_uri('postgresql+psycopg2://postgres:%(username)s@%(hostname)s:%(port)s/%(dbname)s',{'username':alloydb_user,'hostname':hostname,'port':port_num,'dbname':db_name})
    print(db)
    user_input = None

    llm =  VertexAI(
        model_name="text-bison",
        temperature=0.1,
        max_output_tokens=512,
        top_p=0.8,
        top_k=40,
        verbose=True,
        )
    toolkit = SQLDatabaseToolkit(db=db,llm=llm)
    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True
    )
    
    prompt = ("""
            For the following query, if it requires drawing a table, reply as follows:
            {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

            If the query requires creating a bar chart, reply as follows:
            {"bar": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

            If the query requires creating a line chart, reply as follows:
            {"line": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

            There can only be two types of chart, "bar" and "line".

            If it is just asking a question that requires neither, reply as follows:
            {"answer": "answer"}
            Example:
            {"answer": "The title with the highest rating is 'Gilead'"}

            If you do not know the answer, reply as follows:
            {"answer": "I do not know."}

            Return all output as a string.

            All strings in "columns" list and data list, should be in double quotes,

            For example: {"columns": ["title", "ratings_count"], "data": [["Gilead", 361], ["Spider's Web", 5164]]}

            Lets think step by step.

            Below is the query.
            Query: 
            """)

    user_input = st.text_input("Question: ","Find the car with the highest selling price", key="input")
    print(user_input)
    if st.button("Submit"):
        if user_input is None:
            user_input = "Find the car with the highest selling price?"
            result = agent_executor.run("using car_details table, "+ prompt + user_input)
        else:
            result = agent_executor.run("using car_details table, "+ prompt + user_input)
            print(type(result))
        response = None
        response = result
        print(response)
        decoded_response = None
        decoded_response = decode_response(response)
        print("\n\n")
        #print(decoded_response)
        write_response(decoded_response)
            #st.write("\n\n")
            #st.write("## ",result)

def decode_response(response: str) -> dict:
    """This function converts the string response from the model to a dictionary object.

    Args:
        response (str): response from the model

    Returns:
        dict: dictionary with response data
    """
    json_conv = None
    json_conv = ast.literal_eval(response)
    print("Response in JSON: \n",json_conv)
    return json_conv

def write_response(response_dict: dict):
    """
    Write a response from an agent to a Streamlit app.

    Args:
        response_dict: The response from the agent.

    Returns:
        None.
    """
    df = None
    # Check if the response is an answer.
    if "answer" in response_dict:
        st.write("\n\n")
        st.write("##",response_dict["answer"])

    # Check if the response is a bar chart.
    if "bar" in response_dict:
        data = dict(response_dict["bar"])
        print("Dict: \n",response_dict["bar"])
        df = pd.DataFrame(data["data"], columns=data["columns"])
        df.set_index(df.columns[0],inplace=True)
        print("Bar Graph dataframe: \n",df)
        st.write("\n\n")
        st.bar_chart(df,)

    # Check if the response is a line chart.
    if "line" in response_dict:
        data = response_dict["line"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        df.set_index(df.columns[0], inplace=True)
        st.write("\n\n")
        st.line_chart(df)

    # Check if the response is a table.
    if "table" in response_dict:
        data = response_dict["table"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.write("\n\n")
        st.table(df)

def main():
    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.title("One-stop shop to get the right price for your car.")
    st.markdown('---')
    project_id = "raves-altostrat"  # @param {type:"string"}
    region = "asia-southeast1"
    os.environ["GOOGLE_CLOUD_PROJECT"] = project_id

    menu = ["Home","Inventory","Analysis","Vector Similarity","Analytics-Embedding_Based","Architecture","About"]
    choice = st.sidebar.selectbox("Menu",menu)
    st.sidebar.info("Click *Home* for Inventory !!")
    st.sidebar.image("power-by-cloud.png",use_column_width=True,clamp=True)
    if choice == "Home":
        sell_car()
    elif choice =="Inventory":
        inventory()
    elif choice =="Analysis":
        analytics()
    elif choice =="Vector Similarity":
        gen_embeddings()        
    elif choice =="Analytics-Embedding_Based":
        search_embedding()        
    elif choice =="Architecture":
        image = Image.open('AlloyDB_LLM_arch2.png')
        st.image(image,width=1200)

    elif choice =="About":
        i=0
        for i in range(12):
            st.write("")
        st.markdown('## <p style=''font-size:150%;text-align:center;''> Following DEMO is developed and managed by <u>Google Cloud Platform Customer Engineer Team</u> !! </p>',unsafe_allow_html=True)
        st.markdown('---')
        st.balloons()

main()

