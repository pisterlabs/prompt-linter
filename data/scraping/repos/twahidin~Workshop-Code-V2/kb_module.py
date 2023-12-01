import streamlit as st
import sqlite3
import streamlit_antd_components as sac
import pandas as pd
import os
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import LanceDB
from basecode.authenticate import return_api_key
from langchain.docstore.document import Document
import lancedb  
import configparser
import ast
import json

class ConfigHandler:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')

    def get_config_values(self, section, key):
        value = self.config.get(section, key)
        try:
            # Try converting the string value to a Python data structure
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            # If not a data structure, return the plain string
            return value
		
config_handler = ConfigHandler()
TCH = config_handler.get_config_values('constants', 'TCH')
STU = config_handler.get_config_values('constants', 'STU')
SA = config_handler.get_config_values('constants', 'SA')
AD = config_handler.get_config_values('constants', 'AD')



# Create or check for the 'database' directory in the current working directory
cwd = os.getcwd()
WORKING_DIRECTORY = os.path.join(cwd, "database")

if not os.path.exists(WORKING_DIRECTORY):
	os.makedirs(WORKING_DIRECTORY)

if st.secrets["sql_ext_path"] == "None":
	WORKING_DATABASE= os.path.join(WORKING_DIRECTORY , st.secrets["default_db"])
else:
	WORKING_DATABASE= st.secrets["sql_ext_path"]

os.environ["OPENAI_API_KEY"] = return_api_key()
lancedb_path = os.path.join(WORKING_DIRECTORY, "lancedb")
db = lancedb.connect(lancedb_path)


def fetch_vectorstores_with_usernames():
    conn = sqlite3.connect(WORKING_DATABASE)
    cursor = conn.cursor()
    
    query = '''
    SELECT 
        Vector_Stores.vs_id, 
        Subject.subject_name,
        Topic.topic_name, 
        Vector_Stores.vectorstore_name, 
        Users.username, 
        Vector_Stores.sharing_enabled
    FROM Vector_Stores
    JOIN Users ON Vector_Stores.user_id = Users.user_id
    LEFT JOIN Subject ON Vector_Stores.subject = Subject.id
    LEFT JOIN Topic ON Vector_Stores.topic = Topic.id;
    '''
    
    cursor.execute(query)
    data = cursor.fetchall()
    conn.close()
    return data

def display_vectorstores():
    data = fetch_vectorstores_with_usernames()
    df = pd.DataFrame(data, columns=["vs_id", "subject_name", "topic_name", "vectorstore_name", "username", "sharing_enabled"])

    # Convert the 'sharing_enabled' values
    df["sharing_enabled"] = df["sharing_enabled"].apply(lambda x: '✔' if x == 1 else '')

    st.dataframe(
        df, 
        use_container_width=True,
        column_order=["vs_id", "subject_name", "topic_name", "vectorstore_name", "username", "sharing_enabled"]
    )

def fetch_all_files():
    """
    Fetch all files either shared or based on user type
    """
    conn = sqlite3.connect(WORKING_DATABASE)
    cursor = conn.cursor()
    
    # Construct the SQL query with JOINs for Subject, Topic, and Users tables
    if st.session_state.user['profile_id'] == 'SA':
        cursor.execute('''
            SELECT Files.file_id, Files.file_name, Subject.subject_name, Topic.topic_name, Users.username 
            FROM Files 
            JOIN Subject ON Files.subject = Subject.id 
            JOIN Topic ON Files.topic = Topic.id 
            JOIN Users ON Files.user_id = Users.user_id
        ''')
    else:
        cursor.execute('''
            SELECT Files.file_id, Files.file_name, Subject.subject_name, Topic.topic_name, Users.username 
            FROM Files 
            JOIN Subject ON Files.subject = Subject.id 
            JOIN Topic ON Files.topic = Topic.id 
            JOIN Users ON Files.user_id = Users.user_id 
            WHERE Files.sharing_enabled = 1
        ''')
    
    files = cursor.fetchall()
    formatted_files = [f"({file[0]}) {file[1]} ({file[4]})" for file in files]
    
    conn.close()
    return formatted_files


def fetch_file_data(file_id):
    """
    Fetch file data given a file id
    """
    conn = sqlite3.connect(WORKING_DATABASE)
    cursor = conn.cursor()
    
    cursor.execute("SELECT data, metadata FROM Files WHERE file_id = ?", (file_id,))
    data = cursor.fetchone()

    conn.close()
    if data:
        return data[0], data[1]
    else:
        return None, None

def insert_topic(org_id, topic_name):
    conn = sqlite3.connect(WORKING_DATABASE)
    cursor = conn.cursor()

    try:
        cursor.execute('INSERT INTO Topic (org_id, topic_name) VALUES (?, ?);', (org_id, topic_name))
        conn.commit()
        return True  # Indicates successful insertion
    except sqlite3.IntegrityError:
        # IntegrityError occurs if topic_name is not unique within the org
        return False  # Indicates topic_name is not unique within the org
    finally:
        conn.close()

def insert_subject(org_id, subject_name):
    conn = sqlite3.connect(WORKING_DATABASE)
    cursor = conn.cursor()

    try:
        cursor.execute('INSERT INTO Subject (org_id, subject_name) VALUES (?, ?);', (org_id, subject_name))
        conn.commit()
        return True  # Indicates successful insertion
    except sqlite3.IntegrityError:
        # IntegrityError occurs if subject_name is not unique within the org
        return False  # Indicates subject_name is not unique within the org
    finally:
        conn.close()

def select_organization():
    with sqlite3.connect(WORKING_DATABASE) as conn:
        cursor = conn.cursor()

        # Org selection
        org_query = "SELECT org_name FROM Organizations"
        cursor.execute(org_query)
        orgs = cursor.fetchall()
        org_names = [org[0] for org in orgs]

        # Use a Streamlit selectbox to choose an organization
        selected_org_name = st.selectbox("Select an organization:", org_names)

        # Retrieve the org_id for the selected organization
        cursor.execute('SELECT org_id FROM Organizations WHERE org_name = ?;', (selected_org_name,))
        result = cursor.fetchone()

        if result:
            org_id = result[0]
            st.write(f"The org_id for {selected_org_name} is {org_id}.")
            return org_id
        else:
            st.write(f"Organization '{selected_org_name}' not found in the database.")
            return None


def fetch_subjects_by_org(org_id):
    conn = sqlite3.connect(WORKING_DATABASE)
    cursor = conn.cursor()

    # Check if the user is a super_admin (org_id is 0)
    if org_id == 0:
        cursor.execute('SELECT * FROM Subject;')
    else:
        cursor.execute('SELECT * FROM Subject WHERE org_id = ?;', (org_id,))
    
    subjects = cursor.fetchall()
    conn.close()
    return subjects

def fetch_topics_by_org(org_id):
    conn = sqlite3.connect(WORKING_DATABASE)
    cursor = conn.cursor()

    # Check if the user is a super_admin (org_id is 0)
    if org_id == 0:
        cursor.execute('SELECT * FROM Topic;')
    else:
        cursor.execute('SELECT * FROM Topic WHERE org_id = ?;', (org_id,))
    
    topics = cursor.fetchall()
    conn.close()
    return topics

def split_docs(file_path,meta):
#def split_meta_docs(file, source, tch_code):
	loader = UnstructuredFileLoader(file_path)
	documents = loader.load()
	text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
	docs = text_splitter.split_documents(documents)
	metadata = {"source": meta}
	for doc in docs:
		doc.metadata.update(metadata)
	return docs

def create_lancedb_table(embeddings, meta, table_name):
	lancedb_path = os.path.join(WORKING_DIRECTORY, "lancedb")
	# LanceDB connection
	db = lancedb.connect(lancedb_path)
	table = db.create_table(
		f"{table_name}",
		data=[
			{
				"vector": embeddings.embed_query("Query Unsuccessful"),
				"text": "Query Unsuccessful",
				"id": "1",
				"source": f"{meta}"
			}
		],
		mode="overwrite",	
	)
	return table

def save_to_vectorstores(vs, vstore_input_name, subject, topic, username, share_resource=False):
    conn = sqlite3.connect(WORKING_DATABASE)
    cursor = conn.cursor()

    # Fetch the user's details
    cursor.execute('SELECT user_id FROM Users WHERE username = ?', (username,))
    user_details = cursor.fetchone()

    if not user_details:
        st.error("Error: User not found.")
        return

    user_id = user_details[0]

    # If Vector_Store instance exists in session state, then serialize and save
    # vs is the documents in json format and vstore_input_name is the name of the table and vectorstore
    if vs:
        try:
            
            cursor.execute('SELECT 1 FROM Vector_Stores WHERE vectorstore_name LIKE ? AND user_id = ?', (f"%{vstore_input_name}%", user_id))
            exists = cursor.fetchone()

            if exists:
                st.error("Error: An entry with the same vectorstore_name and user_id already exists.")
                return
            
            if subject is None:
                st.error("Error: Subject is missing.")
                return

            if topic is None:
                st.error("Error: Topic is missing.")
                return

            # Get the subject and topic IDs
            cursor.execute('SELECT id FROM Subject WHERE subject_name = ?', (subject,))
            subject_id = cursor.fetchone()[0]

            cursor.execute('SELECT id FROM Topic WHERE topic_name = ?', (topic,))
            topic_id = cursor.fetchone()[0]

            # Insert the new row
            cursor.execute('''
            INSERT INTO Vector_Stores (vectorstore_name, documents, user_id, subject, topic, sharing_enabled)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (vstore_input_name, vs, user_id, subject_id, topic_id, share_resource))

            conn.commit()
            conn.close()
            
            
        except Exception as e:
            st.error(f"Error in storing documents and vectorstore: {e}")
            return

def document_to_dict(doc):
    # Assuming 'doc' has 'page_content' and 'metadata' attributes
    return {
        'page_content': doc.page_content,
        'metadata': doc.metadata
    }

def dict_to_document(doc_dict):
    # Create a Document object from the dictionary
    # Adjust this according to how your Document class is defined
    return Document(page_content=doc_dict['page_content'],metadata=doc_dict['metadata'])

def create_vectorstore():
    openai.api_key = return_api_key()
    os.environ["OPENAI_API_KEY"] = return_api_key()
    full_docs = []
    st.subheader("Enter the topic and subject for your knowledge base")
    embeddings = OpenAIEmbeddings()
    if st.session_state.user['profile_id'] == SA:
        org_id = select_organization()
        if org_id is None:
            return
    else:
        org_id = st.session_state.user["org_id"]
    

    # Fetch all available subjects
    subjects = fetch_subjects_by_org(st.session_state.user["org_id"])
    subject_names = [sub[2] for sub in subjects]  # Assuming index 2 holds the subject_name
    selected_subject = st.selectbox("Select an existing subject or type a new one:", options=subject_names + ['New Subject'])
    
    if selected_subject == 'New Subject':
        subject = st.text_input("Please enter the new subject name:", max_chars=30)
        if subject:
            insert_subject(org_id, subject)

    else:
        subject = selected_subject

    # Fetch all available topics
    topics = fetch_topics_by_org(st.session_state.user["org_id"])
    topic_names = [topic[2] for topic in topics]  # Assuming index 2 holds the topic_name
    selected_topic = st.selectbox("Select an existing topic or type a new one:", options=topic_names + ['New Topic'])
    
    if selected_topic == 'New Topic':
        topic = st.text_input("Please enter the new topic name:", max_chars=30)
        if topic:
            insert_topic(org_id, topic)
    else:
        topic = selected_topic
    
    vectorstore_input = st.text_input("Please type in a name for your knowledge base:", max_chars=20)
    vs_name = vectorstore_input + f"_({st.session_state.user['username']})"
    share_resource = st.checkbox("Share this resource", value=True)  # <-- Added this line

    # Show the current build of files for the latest database
    st.subheader("Select one or more files to build your knowledge base")
    files = fetch_all_files()
    if files:
        selected_files = sac.transfer(items=files, label=None, index=None, titles=['Uploaded files', 'Select files for KB'], format_func='title', width='100%', height=None, search=True, pagination=False, oneway=False, reload=True, disabled=False, return_index=False)
        
        # Alert to confirm the creation of knowledge base
        st.warning("Building your knowledge base will take some time. Please be patient.")
        
        build = sac.buttons([
            dict(label='Build VectorStore', icon='check-circle-fill', color = 'green'),
            dict(label='Cancel', icon='x-circle-fill', color='red'),
        ], label=None, index=1, format_func='title', align='center', position='top', size='default', direction='horizontal', shape='round', type='default', compact=False, return_index=False)
        
        if build == 'Build VectorStore' and selected_files:
            for s_file in selected_files:
                file_id = int(s_file.split("(", 1)[1].split(")", 1)[0])
                file_data, meta = fetch_file_data(file_id)
                docs = split_docs(file_data, meta)
                full_docs.extend(docs)
            #convert full_docs to json to store in sqlite
            full_docs_dicts = [document_to_dict(doc) for doc in full_docs]
            docs_json = json.dumps(full_docs_dicts)
            
            #db = LanceDB.from_documents(full_docs, OpenAIEmbeddings(), connection=create_lancedb_table(embeddings, meta, vs_name))
            #table = create_lancedb_table(embeddings, meta, vs_name)
            # lancedb_path = os.path.join(WORKING_DIRECTORY, "lancedb")
	        # LanceDB connection
            # db = lancedb.connect(lancedb_path)
            # st.session_state.test1 = table
            # st.write("full_docs",full_docs)
            #full_docs_dicts = [document_to_dict(doc) for doc in full_docs]
            #docs_json = json.dumps(full_docs_dicts)
            # st.write("docs_json",docs_json)
            #retrieved_docs_dicts = get_docs()  # Assuming this returns the list of dictionaries
            # retrieved_docs_dicts = json.loads(docs_json)
            # retrieved_docs = [dict_to_document(doc_dict) for doc_dict in retrieved_docs_dicts]
            # st.write("retrieved_docs",retrieved_docs)
            #st.session_state.test2 = json.loads(docs_json)
            # st.session_state.vs = LanceDB.from_documents(retrieved_docs , OpenAIEmbeddings(), connection= db.open_table("_(super_admin)"))
            # st.session_state.current_model = "test1"
            # st.write(st.session_state.test1)
            #st.write(st.session_state.test2)
            #st.write(type(db))
            #st.session_state.vs = load_vectorstore(documents, table_name)
            create_lancedb_table(embeddings, meta, vs_name)
            save_to_vectorstores(docs_json, vs_name, subject, topic, st.session_state.user["username"], share_resource)  # Passing the share_resource to the function
            st.success("Knowledge Base loaded")

    else:
        st.write("No files found in the database.")

def load_vectorstore(documents, table_name):
    
    retrieved_docs_dicts = json.loads(documents)
    retrieved_docs = [dict_to_document(doc_dict) for doc_dict in retrieved_docs_dicts]
    vs = LanceDB.from_documents(retrieved_docs , OpenAIEmbeddings(), connection= db.open_table(f"{table_name}"))
    return vs


def delete_lancedb_table(table_name):
	lancedb_path = os.path.join(WORKING_DIRECTORY, "lancedb")
	# LanceDB connection
	db = lancedb.connect(lancedb_path)
	db.drop_table(f"{table_name}")

def fetch_vectorstores_by_user_id(user_id):
    conn = sqlite3.connect(WORKING_DATABASE)
    cursor = conn.cursor()
    
    # Fetch vectorstores based on user_id
    cursor.execute('SELECT vectorstore_name FROM Vector_Stores WHERE user_id = ?;', (user_id,))
    
    vectorstores = cursor.fetchall()
    conn.close()
    return vectorstores

def delete_vectorstores():
    st.subheader("Delete VectorStores in Database:")
    user_vectorstores = fetch_vectorstores_by_user_id(st.session_state.user["id"])
    
    if user_vectorstores:
        vectorstore_names = [vs[0] for vs in user_vectorstores]
        selected_vectorstores = st.multiselect("Select vectorstores to delete:", options=vectorstore_names)
        confirm_delete = st.checkbox("I understand that this action cannot be undone.", value=False)
        
        if st.button("Delete VectorStore"):
            if confirm_delete and selected_vectorstores:
                delete_vectorstores_from_db(selected_vectorstores, st.session_state.user["id"], st.session_state.user["profile_id"])
                st.success(f"Deleted {len(selected_vectorstores)} vectorstores.")
            else:
                st.warning("Please confirm the deletion action.")
    else:
        st.write("No vectorstores found in the database.")

def delete_vectorstores_from_db(vectorstore_names, user_id, profile):
    conn = sqlite3.connect(WORKING_DATABASE)
    cursor = conn.cursor()

    for vectorstore_name in vectorstore_names:
        if profile in ['SA', 'AD']:
            # Delete the corresponding LanceDB table
            delete_lancedb_table(vectorstore_name)
            
            # Delete vectorstore irrespective of the user_id associated with them
            cursor.execute('DELETE FROM Vector_Stores WHERE vectorstore_name=?;', (vectorstore_name,))
        else:
            # Delete the corresponding LanceDB table
            delete_lancedb_table(vectorstore_name)
            
            # Delete only if the user_id matches
            cursor.execute('DELETE FROM Vector_Stores WHERE vectorstore_name=? AND user_id=?;', (vectorstore_name, user_id))
            
            # Check if the row was affected
            if cursor.rowcount == 0:
                st.error(f"Unable to delete vectorstore '{vectorstore_name}' that is not owned by you.")
                
    conn.commit()  # Commit the changes
    conn.close()  # Close the connection
