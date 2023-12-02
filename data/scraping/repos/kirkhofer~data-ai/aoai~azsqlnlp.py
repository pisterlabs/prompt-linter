import streamlit as st 
import pyodbc
import openai
import requests
import json

openai.api_type = "azure"
openai.api_base = st.secrets.aoai.base
openai.api_key = st.secrets.aoai.key

server = st.secrets.sql.server
database = st.secrets.sql.db
username = st.secrets.sql.user
password = st.secrets.sql.pwd
driver= st.secrets.sql.driver or 'ODBC Driver 18 for SQL Server'

# NOTE: Change this if something isn't working
connStr='DRIVER='+driver+';SERVER=tcp:'+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password+';Encrypt=YES;TrustServerCertificate=YES'

# create a row from array of columns with "ColumnName (Datatype)"
def create_row(columns):
    return ",".join([f"{col['column']} ({col['type']})" for col in columns])

if "tables" not in st.session_state:
    tables=[]
    with pyodbc.connect(connStr) as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT TABLE_SCHEMA,TABLE_NAME,COLUMN_NAME,DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS")
            row = cursor.fetchone()
            while row:
                # tables.append(row)
                tables.append({'schema':row[0],'table':row[1],'column':row[2],'type':row[3]})
                #print (str(row[0]) + " " + str(row[1]))
                row = cursor.fetchone()

    vals = set([(t['schema'],t['table']) for t in tables])

    search=[]
    for val in vals:
        #Get a list of all tables by table and schema
        columns = [c for c in tables if c['table']==val[1] and c['schema']==val[0]]
        cols = create_row(columns)
        search.append(f"{val[0]}.{val[1]} [{cols}]")
    
    st.session_state["tables"]=search


def get_text():
    input_text = st.text_area(height = 200, label="Query here", label_visibility='collapsed', placeholder="Enter query...", key="query_input")
    return input_text

def Get_ChatCompletion(prompt,model="gpt-35-turbo",temperature=0.0):

    st.session_state["prompt"]=prompt
    openai.api_version=st.secrets.aoai.previewversion
    response = openai.ChatCompletion.create(
        engine=model, 
        messages = prompt,
        temperature=temperature,
        max_tokens=1024
    )
    st.session_state["response"]=response
    llm_output = response['choices'][0]['message']['content']
    return llm_output

def Get_Completion(prompt,model="code-davinci-002"):

    openai.api_version=st.secrets.aoai.version
    st.session_state["prompt"]=prompt
    response = openai.Completion.create(
        engine=model, 
        prompt = prompt,
        temperature=0,
        max_tokens=250,
        stop=["#"]
    )
    st.session_state["response"]=response
    llm_output = response.choices[0].text
    return llm_output

def Get_CompletionREST(prompt,max_tokens=250,temperature=0,model="code-davinci-002"):

    headers={"Content-Type":"application/json","api-key":st.secrets.aoai.key}
    uri = f"{st.secrets.aoai.base}openai/deployments/{model}/completions?api-version=2022-12-01"

    body={
        "prompt": prompt,
        "max_tokens":max_tokens,
        "temperature":temperature,
        "stop":["#"]
    }

    #convert body to utf8 bytes
    body_utf8 = bytes(json.dumps(body), 'utf-8')

    st.session_state["prompt"]=body

    request=requests.post(uri, headers=headers, json=body)
    # request=requests.post(uri, headers=headers, data=body_utf8)
    # st.write(f"Status={request.status_code}")

    response=request.json()

    st.session_state["response"]=response
    if( "error" in response ):
        #Read this from this json {"error":{"message":"blah"}}
        return response['error']['message']
    else:
        return response['choices'][0]['text']

models=['Completion','Chat Completion','REST']
runningModel=st.sidebar.selectbox('Model',models)
# st.sidebar.markdown("Enter a query against the tables in the database.  The query will be run against the database and the results will be displayed below.  The query will also be sent to OpenAI to generate a SQL query that will return the same results.  The generated query will be displayed below the results.")
st.sidebar.markdown("## Enter Your query")
st.sidebar.write(f"Running against {database}")
# query_input = get_text()
input_text = st.sidebar.text_area(height = 200, label="Query here", label_visibility='collapsed', placeholder="Enter query...", key="query_input")

if st.sidebar.button("Submit"):  
    st.write(f"## Query:")
    st.write(input_text)
    
    # odelName  `
    #     -temperature 0 -max_tokens 250 -stop "#"
    messages=[]
    lines=[]
    lines.append("Given the following tables and columns with data types:")
    #for each row in tables array combine the two columns and separate by a carriage return
    for t in st.session_state["tables"]:
        lines.append(t)
    
    lines.append(f"#Create a SQL query for {input_text}")
    content="\n".join(lines)
    messages.append({"role":"system","content":content})
    messages.append({"role":"user","content":input_text})
    # st.write(content)
    if "Completion" == runningModel:
        resp = Get_Completion(content)
    elif "Chat Completion" == runningModel:
        resp = Get_ChatCompletion(messages)
    else:
        resp = Get_CompletionREST(content)

    st.write(f"## Results for {runningModel}:")
    st.write(resp)

with st.expander("See the detail"):
    if "response" in st.session_state:
        st.write("response:")
        st.write(st.session_state["response"])    
    if "prompt" in st.session_state:
        st.write("Prompt:")
        st.write(st.session_state["prompt"])    
    st.write(f"Length of tables={len(st.session_state['tables'])}")
    st.table(st.session_state["tables"])

