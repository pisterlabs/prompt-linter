# This code scrapes a website, splits the text into chunks, and embeds them using OpenAI and Pinecone.

from tqdm.auto import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import re
import html
from urllib.parse import urljoin
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import requests
import openai
from bs4 import BeautifulSoup
import sys
import streamlit as st
from myfunc.mojafunkcija import st_style
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_extraction_chain

st_style()


# Define a function to scrape a given URL
def scrape(url: str):
    global headers, sajt, err_log, tiktoken_len, vrsta
    # Send a GET request to the URL
    res = requests.get(url, headers=headers)

    # Check the response status code
    if res.status_code != 200:
        # If the status code is not 200 (OK), write the status code and return None
        err_log += f"{res.status_code} for {url}\n"
        return None

    # If the status code is 200, initialize BeautifulSoup with the response text
    soup = BeautifulSoup(res.text, "html.parser")
    # soup = BeautifulSoup(res.text, 'lxml')

    # Find all links to local pages on the website
    local_links = []
    for link in soup.find_all("a", href=True):
        if (
            link["href"].startswith(sajt)
            or link["href"].startswith("/")
            or link["href"].startswith("./")
        ):
            href = link["href"]
            base_url, extension = os.path.splitext(href)
            if not extension and not "mailto" in href and not "tel" in href:
                local_links.append(urljoin(sajt, href))

                # Find the main content using CSS selectors
                try:
                    # main_content_list = soup.select('body main')
                    main_content_list = soup.select(vrsta)

                    # Check if 'main_content_list' is not empty
                    if main_content_list:
                        main_content = main_content_list[0]

                        # Extract the plaintext of the main content
                        main_content_text = main_content.get_text()

                        # Remove all HTML tags
                        main_content_text = re.sub(r"<[^>]+>", "", main_content_text)

                        # Remove extra white space
                        main_content_text = " ".join(main_content_text.split())

                        # Replace HTML entities with their corresponding characters
                        main_content_text = html.unescape(main_content_text)

                    else:
                        # Handle the case when 'main_content_list' is empty
                        main_content_text = "error"
                        err_log += f"Error in page structure, use body instead\n"
                        st.error(err_log)
                        sys.exit()
                except Exception as e:
                    err_log += f"Error while discovering page content\n"
                    return None

    # return as json
    return {"url": url, "text": main_content_text}, local_links


# Now you can work with the parsed content using Beautiful Soup
def add_schema_data(line):
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Create an instance of ChatOpenAI
    llm = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=openai_api_key)

    # mogu da se definisu bilo koji delovi kao JSON schema

    schema = {
        "properties": {
            "title": {"type": "string"},
            "keyword": {"type": "string"},
        },
        "required": ["title", "keyword"],
    }

    # moze da se ucita bilo koji fajl (ili dokument ili scrapeovan websajt recimo) kao txt ili json
    # chunking treba raditi bez overlapa
    # moze da se razdvoji title i keyword u jedan index,
    # title i text u drugi index, onda imamo i duzi i kraci index

    chain = create_extraction_chain(schema, llm)
    result = chain.run(line)
    for item in result:
        title = item["title"]
        keyword = item["keyword"]

        # ovo treba da postane jedan chunk, na koji se daodaju metadata i onda upsertuje u index
        # prakticno umesto prefix-a tj ovo je dinamicki prefix
        # if title and keyword and text:
        # st.write(f"{title}: keyword: {keyword} ->  {text}\n")
        added_schema_data = f"Title: {title} <- Keyword: {keyword} -> Text: {line}"
        return added_schema_data
    # else:
    #     st.write("No title or keyword or text")


# na kraju se upsertuje u index svaka linija
# opciono moze da se ponovo sacuja u txt ili json fajl


# Define a function to scrape a given URL
def main(chunk_size, chunk_overlap):
    skinuto = False
    napisano = False
    file_name = "chunks.json"
    with st.form(key="my_form_scrape", clear_on_submit=False):
        global res, err_log, headers, sajt, source, vrsta
        st.subheader("Pinecone Scraping")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }

        # Set the domain URL

        # with st.form(key="my_form", clear_on_submit=False):
        sajt = st.text_input("Unesite sajt : ")
        # prefix moze da se definise i dinamicki
        text_prefix = st.text_input(
            "Unesite prefiks za tekst: ",
            help="Prefiks se dodaje na početak teksta pre podela na delove za indeksiranje",
        )
        vrsta = st.radio(
            "Unesite vrstu (default je body main): ", ("body main", "body")
        )
        add_schema = st.radio(
            "Da li želite da dodate Schema Data (može značajno produžiti vreme potrebno za kreiranje): ",
            ("Da", "Ne"),
            help="Schema Data se dodaje na početak teksta",
            key="add_schema_web",
        )
        # chunk_size, chunk_overlap = def_chunk()
        submit_button = st.form_submit_button(label="Submit")
        st.info(f"Chunk veličina: {chunk_size}, chunk preklapanje: {chunk_overlap}")
        if len(text_prefix) > 0:
            text_prefix = text_prefix + " "
        if submit_button and not sajt == "":
            res = requests.get(sajt, headers=headers)
            err_log = ""

            # Read OpenAI API key from file
            openai.api_key = os.environ.get("OPENAI_API_KEY")

            # # Retrieving API keys from files
            # PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

            # # Setting the environment for Pinecone API
            # PINECONE_API_ENV = os.environ.get("PINECONE_API_ENV")

            # pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

            # Initialize BeautifulSoup with the response text
            soup = BeautifulSoup(res.text, "html.parser")
            # soup = BeautifulSoup(res.text, 'html5lib')

            # Define a function to scrape a given URL

            links = [sajt]
            scraped = set()
            data = []
            i = 0
            placeholder = st.empty()

            with st.spinner(f"Scraping "):
                while True:
                    # while i < 2:
                    i += 1
                    if len(links) == 0:
                        st.success("URL lista je kompletirana")
                        break
                    url = links[0]

                    # st.write(f'{url}, ">>", {i}')
                    placeholder.text(f"Obrađujem link broj {i}")
                    try:
                        res = scrape(url)
                        err_log += f" OK scraping {url}: {i}\n"
                    except Exception as e:
                        err_log += f"An error occurred while scraping {url}: page can not be scraped.\n"

                    scraped.add(url)

                    if res is not None:
                        page_content, local_links = res
                        data.append(page_content)
                        # add new links to links list
                        links.extend(local_links)
                        # remove duplicates
                        links = list(set(links))
                    # remove links already scraped
                    links = [link for link in links if link not in scraped]

                # Initialize RecursiveCharacterTextSplitter
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
            txt_string = ""

            progress_text = "Podaci za Embeding se trenutno kreiraju. Molimo sačekajte."

            progress_bar = st.progress(0.0, text=progress_text)
            ph = st.empty()
            progress_barl = st.progress(0.0, text=progress_text)
            ph2 = st.empty()
            ph3 = st.empty()
            # Iterate over data records
            with st.spinner(f"Kreiranje podataka za Embeding"):
                for idx, record in enumerate(tqdm(data)):
                    # Split the text into chunks using the text splitter
                    texts = text_splitter.split_text(record["text"])

                    sto = len(data)
                    odsto = idx + 1
                    procenat = odsto / sto

                    k = int(odsto / sto * 100)
                    progress_bar.progress(procenat, text=progress_text)
                    ph.text(f"Učitano {odsto} od {sto} linkova što je {k} % ")
                    # Create a list of chunks for each text

                    # ovde moze da se doda dinamicko dadavanje prefixa
                    for il in range(len(texts)):
                        stol = len(texts)
                        odstol = il + 1
                        procenatl = odstol / stol

                        kl = int(odstol / stol * 100)
                        progress_barl.progress(procenatl, text=progress_text)
                        ph2.text(f"Učitano {odstol} od {stol} chunkova što je {kl} % ")

                        try:
                            if add_schema == "Da":
                                texts[il] = add_schema_data(texts[il])

                                with st.expander(
                                    f"Obrađeni tekst, link: {odsto} deo: {odstol}"
                                ):
                                    st.write(texts[il])
                        except Exception as e:
                            st.error("Prefiks nije na raspolaganju za ovaj chunk.")

                        # Loop through the Document objects and convert them to JSON

                        # # Specify the file name where you want to save the data
                        content = text_prefix + texts[il]
                        txt_string += content.replace("\n", "") + "\n"

            napisano = st.info(
                "Tekstovi su sačuvani u TXT obliku, downloadujte ih na svoj računar"
            )

            # Specify the file name where you want to save the JSON data

    parsed_url = urlparse(sajt)
    # Get the netloc (which includes the website name)
    website_name = parsed_url.netloc
    # Remove any potential "www." prefix
    if website_name.startswith("www."):
        website_name = website_name[4:]
    parts = website_name.split(".")
    if len(parts) > 1:
        website_name = parts[0]

    if napisano:
        skinuto = st.download_button(
            "Download TXT",
            txt_string,
            file_name=f"hybrid_{website_name}.txt",
        )
    if skinuto:
        st.success(f"Tekstovi sačuvani na {file_name} su sada spremni za Embeding")
