#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json
import streamlit as st
from dotenv import load_dotenv, set_key
import pandas as pd
import os
import io
import zipfile
import csv
import openai
from bs4 import BeautifulSoup
from PIL import Image
import pyscreenshot as ImageGrab


st.set_page_config(
    page_title="𝖲𝖾𝖼𝗎𝗋𝗂𝗍𝗒𝖲𝖼𝗈𝗋𝖾𝖼𝖺𝗋𝖽 𝖠𝗍𝗍𝖺𝖼𝗄 𝖲𝗎𝗋𝖿𝖺𝖼𝖾 𝖨𝗇𝗍𝖾𝗅𝗅𝗂𝗀𝖾𝗇𝖼𝖾 𝖠𝖯𝖨",
    page_icon="https://github.com/securityscorecard/ssc-asi-tools/raw/dev/res/images/SSC.Ti.ANSI.48x48.png",
    layout="wide",
    initial_sidebar_state="expanded",
)


load_dotenv(".env")

api_key = os.environ.get("ASI_TOKEN")
openai.api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    api_key = st.text_input("Enter ASI_TOKEN API key")
    set_key(".env", "ASI_TOKEN", api_key)

if not openai.api_key:
    openai.api_key = st.text_input("Enter OPENAI_API_KEY API key")
    set_key(".env", "OPENAI_API_KEY", openai.api_key)

os.environ["OPENAI_API_KEY"] = openai.api_key
os.environ["ASI_TOKEN"] = api_key

default_temperature = 1.0
temperature = default_temperature
personas = "ssc_personas"
query_persona = "ASIQuery"
default_persona = "ThreatHunter"
with open(os.path.join(personas, f"{query_persona}.txt"), "r") as f:
    persona_text = f.read()

search_url = "https://api.securityscorecard.io/asi/search"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Token " + str(api_key),
}


def search_assets(query):
    data = {"query": query, "cursor": "initial", "size": 1000}
    response = requests.post(search_url, json=data, headers=headers).json()
    return response


def get_ip_info(ip_address):
    data = {
        "query": query,
        "sortDir": "asc",
        "page": 5,
        "index": "ipv4",
        "parser": "structured",
        "size": 1000,
    }
    response = requests.post(search_url, json=data, headers=headers).json()
    return response


def search_leaked_credentials(query):
    data = {"query": query, "index": "leakedcreds", "parser": "structured"}
    response = requests.post(search_url, json=data, headers=headers).json()
    return response


def parse_html_to_text(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    return " ".join(soup.stripped_strings)


prebuilt_queries = [
    "(and port:22 threat_actor:'Lazarus Group' industry:'GOVERNMENT')",
    "(and has_ransomware:1 industry:'FINANCIAL_SERVICES' country:'DE')",
    "(and has_infection:1 (and (or service:'dnp3' service:'modbus'))) ",
    "(and crawling_detected_library_name: 'WordPress' (and (or hostname:'dev' hostname:'test' hostname:'qa' hostname:'staging'))) ",
    "(and org:'SecurityScorecard' cloud_provider:'aws' cloud_region:'us-east-1')",
    "(and has_cve:1 has_scorecard:0 os_type:'Windows')",
    "(and os_type:'FortiOS' (not (or cloud_provider:'aws' cloud_provider:'oracle' cloud_provider:'gce' cloud_provider:'azure')))",
    "(and country:'US' ssl_subject_country:'CN' device_type:'webcam')",
    "(and has_malrep:1 ransomware_group:'CONTI')",
    "(and has_infection:1 has_threatactor:1 has_cve:1 has_ransomware:1 has_malrep:1)",
]

col1, col2 = st.columns([1, 4])
with col1:
    st.image(
        "https://securityscorecard.com/wp-content/uploads/2023/02/android-chrome-512x512-1.png",
        width=100,
    )
with col2:
    st.header("𝖲𝖾𝖼𝗎𝗋𝗂𝗍𝗒𝖲𝖼𝗈𝗋𝖾𝖼𝖺𝗋𝖽 𝖠𝗍𝗍𝖺𝖼𝗄 𝖲𝗎𝗋𝖿𝖺𝖼𝖾 𝖨𝗇𝗍𝖾𝗅𝗅𝗂𝗀𝖾𝗇𝖼𝖾 𝖠𝖯𝖨")

st.sidebar.image(
    "https://raw.githubusercontent.com/NoDataFound/sscGPT/master/tools/sscGPT/images/ssc_logo.png"
)


logo_col, query_col, button_col = st.sidebar.columns([4, 5, 2])


with logo_col:
    asi_enpoints = ["All Assets", "LeakedCreds", "Prebuilt", "File Upload","ASI Query from URL"]
    search_type = st.selectbox("", asi_enpoints,  label_visibility="hidden")


with query_col:
    query = st.text_input("", placeholder="Enter search query", label_visibility="hidden")
    st.markdown(
        """
        <style>

        input[type="text"] {
            width: 100%;
            padding: 7px 5px;
            margin: 0.5px ;
            box-sizing: border-box;
            border: .5px solid #673ff2;
            border-radius: 2px;

        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

st.write(
    "<p style='text-align: justify; font-size: small; color: gray; font-style: italic;'>\
<strong>Disclaimer:</strong> The information provided with this application is 'as-is' and you acknowledge and agree that SecurityScorecard, Inc. makes no representation or warranty, express or implied, as to the accuracy or completeness of the information. You agree that SecurityScorecard shall not have any liability resulting from your use of this information or reliance on the same.\
</p>",
    unsafe_allow_html=True,
)
textdir = "output/Text/" + query.replace(" ", "_").rstrip("\r\n") + ".txt"
jsondir = "output/JSON/" + query.replace(" ", "_").rstrip("\r\n") + ".json"
csvdir = "output/CSV/" + query.replace(" ", "_").rstrip("\r\n") + ".csv"

if query != "":
    show_csv = st.sidebar.checkbox("Show CSV")
    st.markdown(
        """
        <style>
        input[type="text"] {
            width: 100%;
            padding: 8px 5px;
            margin: 1px ;
            box-sizing: border-box;
            border: .5px solid #673ff2;
            border-radius: 2px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    try:
        df = pd.read_csv(csvdir)
    except FileNotFoundError:
        st.warning("No CSV file found for the given query.")
        df = pd.DataFrame()

    if not df.empty and show_csv:
        search_field = st.text_input("Enter field name to search in CSV results")
        row_display_count = st.slider(
            "Number of rows to display", 1, len(df), len(df), 1
        )
        st.write(
            '<style>div[data-baseweb="table"] table { width: 100% !important; }</style>',
            unsafe_allow_html=True,
        )
        table = st.dataframe(df.head(row_display_count), height=600)

        filtered_data = df.to_csv(index=False).encode("utf-8")
        container = st.empty()

        if search_field:
            search_field = search_field.lower()
            df = df.apply(lambda x: x.astype(str).str.lower())
            df = df[df.apply(lambda row: search_field in row.values, axis=1)]
            df = df.apply(lambda x: x.astype(str).str.title())

        if container.button("Download Filtered Data", key="download_button"):
            df_flat = pd.DataFrame(df.stack(), columns=["value"]).reset_index()
            filtered_data = df_flat.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Filtered Data",
                data=filtered_data,
                file_name="filtered_data.csv",
                mime="text/csv",
            )
    elif not df.empty and not show_csv:
        st.info("CSV table is hidden. Check the 'Show CSV' checkbox to display it.")
    else:
        st.warning("No data to display.")

with st.expander("Quick Start", expanded=False):
    st.markdown(
        """
    `TLDR (Quickstart)`

    1. Enter Search Query or select from list of pre built examples and press search.

    2. View or Download JSON or CSV Results
        - Check `Show CSV` to display data in page

    3. Select Analysis Persona, set Temperature and press "Generate SSCGPT Results"
        - Search results will be sent to OpenAI and a report will be generated based on persona type"""
    )
    st.write(
        """
Attack Surface Intelligence API Query guide: https://support.securityscorecard.com/hc/en-us/articles/7659237759515-Create-your-own-ASI-queries#h_01GAJMX665W3S871DBJ1C6QCK7

SecurityScorecard Research blog: https://securityscorecard.com/blog?category=research"""
    )
    st.markdown("---")
with open(os.path.join(personas, f"{query_persona}.txt"), "r") as f:
    persona_text = f.read()
    if search_type == "Prebuilt":

        ssclogo_col, sscquery_col = st.sidebar.columns([1, 10])

        with ssclogo_col:
            st.image("https://simpleicons.org/icons/securityscorecard.svg", width=50)

        with sscquery_col:
            st.info("Select a prebuilt query")
        query_option = st.sidebar.selectbox("", prebuilt_queries, label_visibility="hidden")
    
    asi_persona = personas + "/ASIQuery.txt"
    with open(asi_persona, "r") as f:
            content = f.read()
            persona_asi = personas + content
            

    if search_type == "ASI Query from URL":
        url = st.sidebar.text_input("", placeholder="Enter URL and press enter")
        #send_to_asi = st.sidebar.checkbox("Check this box to send search to ASI")
        generated_text_chunks = []
        if url:

            try:
                chunk_size = 2800
                response = requests.get(url)
                response.raise_for_status()
                parsed_text = parse_html_to_text(response.text)

                num_chunks = len(parsed_text) // chunk_size + (len(parsed_text) % chunk_size > 0)
                st.warning(f"`{num_chunks}` x `{chunk_size}` token (word) packages will be submitted to OpenAI model: `text-davinci-003`") 
                for i in range(0, len(parsed_text), chunk_size):
                    chunk = parsed_text[i:i+chunk_size]
                    chunk_length = len(chunk)
                    prompt_template = "Read contents of {}, parse for indicators and use as data {}. Do not print search results."
                    prompt = prompt_template.format(chunk, persona_asi)
                    completions = openai.Completion.create(
                        engine="text-davinci-003",
                        prompt=prompt,
                        max_tokens=2024,
                        n=1,
                        stop=None,
                        temperature=1.0,
                    )
                    generated_text_chunks.append(completions.choices[0].text.strip())    
                total_size =  num_chunks * chunk_size
                col1, col2, col3 = st.columns(3)
                col1.metric("HTML Word Count", total_size,total_size ) 
                col2.metric("Token Packages", num_chunks,num_chunks ) 
                col3.metric("Total Prompt Token Count", len(prompt), len(prompt))
                st.markdown("----")
                st.info("Generated Attack Surface Intelligence Query from URL")


            except requests.exceptions.RequestException as e:
                st.error(f"Error occurred while fetching the URL: {e}")

            generated_text = '\n'.join(generated_text_chunks)
            query = completions.choices[0].text.strip()
            if st.checkbox("Check this box to send search to ASI"):
                query_str = generated_text_chunks[0].split(":")[1].strip() # Extract the query string
                #st.write(f"Query: {query_str}")
                assets = search_assets(query_str)
                st.write(assets)
            st.write(f"{generated_text_chunks[0]}")    

                    #st.write(total_size)  
                    #st.write(generated_text)
                    #st.write(generated_text_chunks)
            
        
        results = search_assets(query) 
        ssclogo_col, sscquery_col = st.sidebar.columns([1, 10])
        with ssclogo_col:
            st.image("https://simpleicons.org/icons/securityscorecard.svg", width=50)
        with sscquery_col:
            st.info("Generate Attack Surface Intelligence Query from URL")
    
    if search_type == "All Assets":

        sscassetlogo_col, sscassetquery_col = st.sidebar.columns([1, 10])

        with sscassetlogo_col:
            st.image("https://simpleicons.org/icons/securityscorecard.svg", width=50)

        with sscassetquery_col:
            st.info("Search All ASI facets")
    if search_type == "LeakedCreds":
        sscassetlogo_col, sscassetquery_col = st.sidebar.columns([1, 10])

        with sscassetlogo_col:
            st.image("https://simpleicons.org/icons/securityscorecard.svg", width=50)

        with sscassetquery_col:
            st.warning("Search ASI for LeakedCreds")
    if search_type == "File Upload":

        file = st.sidebar.file_uploader("Choose a file")
        if file is not None:
            content = file.readlines()
            #content_lines = content.split("\n")

        sscassetlogo_col, sscassetquery_col = st.sidebar.columns([1, 10])

        with sscassetlogo_col:
            st.image("https://simpleicons.org/icons/securityscorecard.svg", width=50)

        with sscassetquery_col:
            st.warning("Read File contents and search ASI for each line")
    with button_col:
        st.write("")
        st.write("")
        if st.button("𝖲𝖾𝖺𝗋𝖼𝗁"):
            pass
    if search_type == "All Assets":
        results = search_assets(query)

        st.json(results)
        with open(jsondir, "w", encoding="UTF-8") as jsonout:
            json.dump(results, jsonout)
        df = pd.DataFrame(results["hits"])
        df.to_csv(csvdir, index=False)
        with open(textdir, "w", encoding="UTF-8") as textout:
            for hit in results["hits"]:
                for key, value in hit.items():
                    textout.write(f"{key}: {value}\n")
                    textout.write("\n")
        with open(jsondir, "r", encoding="UTF-8") as file:
            json_content = file.read()
        json_button = st.download_button(
            label="Download JSON",
            data=json_content,
            file_name=f"{query.replace(' ', '_')}.json",
            mime="application/json",
        )
        with open(csvdir, "r", encoding="UTF-8") as file:
            csv_content = file.read()
        csv_button = st.download_button(
            label="Download CSV",
            data=csv_content,
            file_name=f"{query.replace(' ', '_')}.csv",
            mime="text/csv",
        )
    elif search_type == "LeakedCreds":
            results = search_leaked_credentials(query)
            st.json(results)
            with open(jsondir, "w", encoding="UTF-8") as jsonout:
                json.dump(results, jsonout)
            df = pd.DataFrame(results["hits"])
            df.to_csv(csvdir, index=False)
            with open(textdir, "w", encoding="UTF-8") as textout:
                for hit in results["hits"]:
                    for key, value in hit.items():
                        textout.write(f"{key}: {value}\n")
                        textout.write("\n")
            with open(jsondir, "r", encoding="UTF-8") as file:

                json_content = file.read()
            json_button = st.download_button(
                label="Download JSON",
                data=json_content,
                file_name=f"{query.replace(' ', '_')}.json",
                mime="application/json",
            )

            with open(csvdir, "r", encoding="UTF-8") as file:
                csv_content = file.read()
            csv_button = st.download_button(
                label="Download CSV",
                data=csv_content,
                file_name=f"{query.replace(' ', '_')}.csv",
                mime="text/csv",
            )    
    elif search_type == "Prebuilt":
            results = search_assets(query_option)
            st.json(results)
            with open(jsondir, "w", encoding="UTF-8") as jsonout:
                json.dump(results, jsonout)
            df = pd.DataFrame(results["hits"])
            df.to_csv(csvdir, index=False)
            with open(textdir, "w", encoding="UTF-8") as textout:
                for hit in results["hits"]:
                    for key, value in hit.items():
                        textout.write(f"{key}: {value}\n")
                        textout.write("\n")

            with open(jsondir, "r", encoding="UTF-8") as file:
                json_content = file.read()
            json_button = st.download_button(
                label="Download JSON",
                data=json_content,
                file_name=f"{query.replace(' ', '_')}.json",
                mime="application/json",
            )

            with open(csvdir, "r", encoding="UTF-8") as file:
                csv_content = file.read()
            csv_button = st.download_button(
                label="Download CSV",
                data=csv_content,
                file_name=f"{query.replace(' ', '_')}.csv",
                mime="text/csv",
            )
    elif search_type == "File Upload":
            results = search_assets(query)
            for line in content:
                line = line.strip()
                for chunk in line:

                    with open(textdir, "r") as textfile:
                        st.json(results)
                    with open(jsondir, "w", encoding="UTF-8") as jsonout:
                        json.dump(results, jsonout)
                    df = pd.DataFrame(results["hits"])
                    df.to_csv(csvdir, index=False)
                    with open(textdir, "w", encoding="UTF-8") as textout:
                        for hit in results["hits"]:
                            for key, value in hit.items():
                                textout.write(f"{key}: {value}\n")
                                textout.write("\n")
                    with open(jsondir, "r", encoding="UTF-8") as file:
                        json_content = file.read()

                    json_button = st.download_button(
                        label="Download JSON",
                        data=json_content,
                        file_name=f"{query.replace(' ', '_')}.json",
                        mime="application/json",
                    )
                    with open(csvdir, "r", encoding="UTF-8") as file:
                        csv_content = file.read()

                    csv_button = st.download_button(
                        label="Download CSV",
                        data=csv_content,
                        file_name=f"{query.replace(' ', '_')}.csv",
                        mime="text/csv",
                    )    
persona_files = [f.split(".")[0] for f in os.listdir(personas) if f.endswith(".txt")]

st.sidebar.markdown("----")

def get_persona_files():
    return [f.split(".")[0] for f in os.listdir(personas) if f.endswith(".txt")]

persona_files = get_persona_files()
ssclogo_col, sscquery_col = st.sidebar.columns([1, 10])

with ssclogo_col:
    st.image("https://simpleicons.org/icons/securityscorecard.svg", width=50)
with sscquery_col:
    st.error("👤 Select Persona")

selected_persona = st.sidebar.selectbox("", [default_persona] + persona_files)

st.sidebar.markdown("----")
ologo_col, oquery_col = st.sidebar.columns([1, 10])

with ologo_col:
    st.image("https://simpleicons.org/icons/openai.svg", width=50)

with oquery_col:
    st.info("sscGPT analysis")
    sscpgt_button = st.sidebar.button("Generate sscGPT analysis")

if sscpgt_button == 1:
    with open(os.path.join(personas, f"{selected_persona}.txt"), "r") as f:
        persona_text = f.read()
    with open(textdir, "r") as textfile:
        for line in textfile:
            line = line.strip()
            if ":" in line:
                text_text = line.split(":", 1)[1].strip()
                if text_text != "" and text_text != [""]:
                    data = textfile.read()
                    input_chunks = [
                        data[i : i + 2500] for i in range(0, len(data), 2500)
                    ]
                generated_text_chunks = []
                for chunk in input_chunks:
                    response = openai.Completion.create(
                        engine="text-davinci-003",
                        prompt=f"data = {chunk} {persona_text} do not print {chunk} directly. ",
                        max_tokens=1024,
                        n=1,
                        stop=None,
                        temperature=temperature,
                    )
                    generated_text_chunks.append(response.choices[0].text.strip())
                    generated_text = "\n".join(generated_text_chunks)
                    break

    st.subheader(
        "𝖲𝖾𝖼𝗎𝗋𝗂𝗍𝗒𝖲𝖼𝗈𝗋𝖾𝖼𝖺𝗋𝖽 𝖠𝗍𝗍𝖺𝖼𝗄 𝖲𝗎𝗋𝖿𝖺𝖼𝖾 𝖨𝗇𝗍𝖾𝗅𝗅𝗂𝗀𝖾𝗇𝖼𝖾 chatGPT Analysis Results"
    )
    st.write(generated_text)

st.sidebar.markdown("----")
ssclogo_col, sscquery_col = st.sidebar.columns([1, 10])

with ssclogo_col:
    st.image("https://simpleicons.org/icons/securityscorecard.svg", width=24)
with sscquery_col:
    st.caption("Manage Personas")
expand_section = st.sidebar.expander("👤 Manage Personas", expanded=False)


with expand_section:
    if selected_persona:
        with open(os.path.join(personas, f"{selected_persona}.txt"), "r") as f:
            persona_text = f.read()
        new_persona_name = st.text_input("Persona Name:", value=selected_persona)
        new_persona_prompt = st.text_area(
            "Persona Prompt:", value=persona_text, height=100
        )

        if new_persona_name != selected_persona or new_persona_prompt != persona_text:
            with open(os.path.join(personas, f"{new_persona_name}.txt"), "w") as f:
                f.write(new_persona_prompt)

            if new_persona_name != selected_persona:
                os.remove(os.path.join(personas, f"{selected_persona}.txt"))
                persona_files.remove(selected_persona)
                persona_files.append(new_persona_name)
                selected_persona = new_persona_name

        if st.button("➖ Delete Persona"):
            if st.warning("Persona Deleted"):
                os.remove(os.path.join(personas, f"{selected_persona}.txt"))
                persona_files.remove(selected_persona)
                selected_persona = ""
expand_section = st.sidebar.expander("➕ Add new Persona", expanded=False)
with expand_section:
    st.subheader("➕ Add new Persona")
    st.text("Press enter to update/save")

    persona_files = get_persona_files()
    new_persona_name = st.text_input("Persona Name:")
    if new_persona_name in persona_files:
        st.error("This persona name already exists. Please choose a different name.")
    else:
        new_persona_prompt = st.text_area("Persona Prompt:", height=100)
        if new_persona_name and new_persona_prompt:
            with open(os.path.join(personas, f"{new_persona_name}.txt"), "w") as f:
                f.write(new_persona_prompt)
            persona_files.append(new_persona_name)
            selected_persona = new_persona_name
    if selected_persona:
        with open(os.path.join(personas, f"{selected_persona}.txt"), "r") as f:
            persona_text = f.read()
    st.text("Press Enter to add")

expand_section = st.expander(
    "Generate a Search Query from URL",
    expanded=False,
)
