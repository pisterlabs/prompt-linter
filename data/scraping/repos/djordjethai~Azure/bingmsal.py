import json
import streamlit as st
from azure.storage.blob import BlobServiceClient
import pandas as pd
from streamlit_javascript import st_javascript
import os
import pdfkit

rights = ""
# Function to fetch AAD username using JavaScript
# filename = "data.json"
filename = "prompt_turbo_miljan.txt"
container_name = "positive-user"
constr = os.environ.get("AZ_BLOB_API_KEY")


from langchain.agents.agent_toolkits import O365Toolkit


toolkit = O365Toolkit()
st.write(toolkit.account)
tools = toolkit.get_tools()
st.write(tools)

from langchain.llms import OpenAI
from langchain.agents import initialize_agent, AgentType

llm = OpenAI(temperature=0)
agent = initialize_agent(
    tools=toolkit.get_tools(),
    llm=llm,
    verbose=False,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
)

agent.run(
     "Create an email draft for me to edit explaining that i am currently in Bangkok. Under no circumstances may you send the message, however."
)
st.write("gotovo")
# def load_data():
#     try:
#         blob_service_client = BlobServiceClient.from_connection_string(constr)
#         container_client = blob_service_client.get_container_client(
#             container_name)
#         blob_client = container_client.get_blob_client(filename)

#         streamdownloader = blob_client.download_blob()
#         data = json.loads(streamdownloader.readall())
#         return data
#     except FileNotFoundError:
#         return {"user_list": []}


st.subheader("Edit System prompta za Miljan Bot-a ")
st.caption("Unesite tekst koji zelite da se prikaze kao sistemski prompt za Miljan Bot-a. Mozete menjati velicinu prozora sa drag dole desno. Sistemski prompt je tekst koji se prikazuje u chatu pre nego sto korisnik zapocne razgovor sa botom Korigujte tekst po zelji i pritisnite Sacuvaj")
st.caption("Ver.30.09.23")


def load_data():
    try:
        blob_service_client = BlobServiceClient.from_connection_string(constr)
        container_client = blob_service_client.get_container_client(
            container_name)
        blob_client = container_client.get_blob_client(filename)

        streamdownloader = blob_client.download_blob()
        data = streamdownloader.readall().decode("utf-8")
        return data
    except FileNotFoundError:
        return {"Nisam pronasao fajl"}


# st.write (load_data)


# def read_aad_username():
#     js_code = """(await fetch("/.auth/me")
#         .then(function(response) {return response.json();}).then(function(body) {return body;}))
#     """

#     return_value = st_javascript(js_code)

#     username = None
#     if return_value == 0:
#         pass  # this is the result before the actual value is returned
#     elif isinstance(return_value, list) and len(return_value) > 0:  # this is the actual value
#         username = return_value[0]["user_id"]
#         st.write(f"Logged in as {username}")
#     else:
#         st.warning(
#             f"could not directly read username from azure active directory: {return_value}.")  # this is an error
#         st.warning(
#             f"A workaround to this is to clear your browser cookies for this site and reloading it.")
#     return username


# # Get the current user's username
# current_user = read_aad_username()
# phtable = st.empty()
# st.info(
#     f"Current User is: {current_user} ")


# # Read data from JSON file
data = load_data()

with st.form(key="my_form"):
    novi_text = st.text_area("Izmenite system prompt:", data, height=300)
    sacuvaj = st.form_submit_button("Sacuvaj")
if sacuvaj:
    try:
        blob_service_client = BlobServiceClient.from_connection_string(constr)
        container_client = blob_service_client.get_container_client(
            container_name)
        blob_client = container_client.get_blob_client(filename)
        blob_client.upload_blob(novi_text, overwrite=True)
        st.success("Uspesno sacuvano")
    except Exception as e:
        st.error(f"Greska pri cuvanju, pokusajte ponovo {e}")




# with phtable.container():
#     st.subheader("User list: ")
#     df = pd.DataFrame(data['user_list'])
#     st.table(df)


# # Access the user list from the JSON dictionary
# user_list = data['user_list']
# nasao = False

# # current_user = "djordje@positive.rs" # for testing purposes

# for user in user_list:
#     if user["name"] == current_user:
#         rights = user["rights"]
#         # Action if the user is found in the list
#         st.success(
#             f"User found in the list and has {rights} rights to use this part of the application")
#         # You can call a function here to perform some action
#         nasao = True

# if not nasao:
#     st.warning("User not found in the list")
#     # The user is not authorized to use this part of the application

# st.subheader("Demo PDF generation")
# html = '<meta charset="UTF-8">' + '<h1>Hello, World!</h1><p>Augmented Intelligence (AI) offers incredible ščđć for businesses. Yet, they are met with several challenges while implementing of AI supported solutions. Here, we dissect the main hurdles. Moreover, we provide the prerequisites needed to successfully integrate AI within a business context.</p>'
# options = {
#         'encoding': 'UTF-8',  # Set the encoding to UTF-8
#         'no-outline': None,
#         'quiet': ''
#     }
# try:
#     pdf_data = pdfkit.from_string(html, cover_first=False, options=options)

#     st.download_button(label="Download Zapisnik as .pdf",
#                                    data=pdf_data,
#                                    file_name="output.pdf",
#                                    mime='application/octet-stream')
# except:
#     st.info("PDF not generated, please try again in 5 minutes")
# # The data.json file can contain additional elements such as access rights, etc.
# # Additionally, you can store data.json in an Azure Blob to avoid redeployment on data changes.


