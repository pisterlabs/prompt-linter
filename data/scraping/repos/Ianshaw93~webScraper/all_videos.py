import json
import os

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"] 
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_ENV = os.environ["PINECONE_ENV"]
# ytd-rich-item-renderer
# ytd-rich-grid-media
base_url = 'https://www.youtube.com/@TheKneesovertoesguy/videos'
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from langchain.document_loaders import YoutubeLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from shared_funcs import recursive_text_splitter
from langchain.vectorstores import Pinecone


def get_video_metadata(url):
    loader = YoutubeLoader.from_youtube_url(url, 
                                            add_video_info=True, 
                                            translation="en",
                                            )
    result = loader.load()
    for video in result:
        video.metadata['link'] = url
        # should an index be added here?
    
    # split text, maintain metadata
    doc = result[0]
    docs = recursive_text_splitter.create_documents([doc.page_content], metadatas=[doc.metadata])
    return docs

def collate_yt_video_links(base_url):
    driver = webdriver.Chrome()
    driver.get(base_url)

    try:
        # Adjust the selector based on the actual structure of the consent form
        accept_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[@aria-label='Reject all']"))
        )
        accept_button.click()
    except:
        print("Cookie consent form not found or failed to click the button.")

    videos_loaded = False
    data_array = []

    while not videos_loaded:
        WebDriverWait(driver, 10)

        # Get the page source after JavaScript execution
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser').find_all('a')
        links = [link for link in soup if link.get('href') and link.get('href').startswith('/watch?v=')]
        if links:
            videos_loaded = True
        # remove duplicates
            links = list(set(links))
    yt_video_links = ['https://www.youtube.com' + f.get('href') for f in links]
    docs_list = []
    for idx, link in enumerate(yt_video_links):
        for doc in get_video_metadata(link):
            docs_list.append(doc)

    seen_content = set()
    filtered_docs_list = []

    for doc in docs_list:
        if doc.page_content not in seen_content:
            seen_content.add(doc.page_content)
            filtered_docs_list.append(doc)

    docs_list = filtered_docs_list
    
    embeddings = OpenAIEmbeddings()


    import pinecone

    # initialize pinecone
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
        environment=os.getenv("PINECONE_ENV"),  # next to api key in console
    )

    # can further indexes be created on free tier?
    index_name = "kot-youtube"

    # First, check if our index already exists. If it doesn't, we create it
    # if index_name not in pinecone.list_indexes():
        # we create a new index
    pinecone.create_index(
    name=index_name,
    metric='cosine',
    dimension=1536  
    )
    # The OpenAI embedding model `text-embedding-ada-002 uses 1536 dimensions`
    docsearch = Pinecone.from_documents(docs_list, embeddings, index_name=index_name)


    # if you already have an index, you can load it like this
    # docsearch = Pinecone.from_existing_index(index_name, embeddings)

    query = "how to train the tibialis anterior without training experience?"
    docs = docsearch.similarity_search(query)
    # issue - whole document returned, not just the relevant section

    # QUERY LLM
    from langchain.prompts import PromptTemplate

    template = """
        Use only the information in the context to answer the question.
        Context: {context}
        Question: {question}
        If the answer is not in the context, DO NOT MAKE UP AN ANSWER.

    """
    prompt = PromptTemplate.from_template(
            template
        )

    # TODO: include text of all relevant docs
    # TODO: remove index from db and make docs smaller
    all_context = ('').join([f.page_content for f in docs])
    final_prompt = prompt.format(context=all_context ,question=query)

    # TODO: send prompt to chatgpt
    from langchain.llms import OpenAI
    # use only one source for query?
    # later split into smaller
    llm = OpenAI(model_name="text-ada-001", openai_api_key=OPENAI_API_KEY)
    answer = llm(final_prompt)
    pass


    # with open('video_transcripts.txt', 'w') as f:
    #     f.write(docs)
    #     json.dump(docs, f)

    # video_links = driver.find_element(By.CSS_SELECTOR, 'a[id="thumbnail"]')
    # # response = requests.get(base_url)
    # yt_video_links = []
    # for link in soup.find_all('a'):
    #     # if '/watch?v=' in link:
    #     if link.get('href') and link.get('href').startswith('/watch?v='):
    #         yt_video_links.append('https://www.youtube.com' + link.get('href'))
    return yt_video_links
# <form action="https://consent.youtube.com/save" method="POST" style="display:inline;" jsaction="JIbuQc:fN3dRc(tWT92d)">
# [article.find('a', href=True)['href'].split('?')[0] for article in articles if article.find('a', href=True)]
# /html/body/ytd-app/div[1]/ytd-page-manager/ytd-browse/ytd-two-column-browse-results-renderer/div[1]/ytd-rich-grid-renderer/div[7]/ytd-rich-grid-row[1]/div/ytd-rich-item-renderer/div/ytd-rich-grid-media/div[1]/div[1]/ytd-thumbnail/a
# <a id="thumbnail" class="yt-simple-endpoint inline-block style-scope ytd-thumbnail" aria-hidden="true" tabindex="-1" rel="null" href="/watch?v=l8Bbmn1Osyo">

vid_links = collate_yt_video_links(base_url)
pass