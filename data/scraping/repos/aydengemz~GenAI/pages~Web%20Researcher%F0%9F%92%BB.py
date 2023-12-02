from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
import os
import openai
from dotenv import find_dotenv, load_dotenv
import requests
import json
import streamlit as st


SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")


def search(query, sKey):
    url = "https://google.serper.dev/search"

    payload = json.dumps({"q": query})
    headers = {"X-API-KEY": sKey, "Content-Type": "application/json"}

    response = requests.request("POST", url, headers=headers, data=payload)
    response_data = response.json()

    print("search results: ", response_data)
    return response_data


def find_best_article_urls(response_data, query):
    response_str = json.dumps(response_data)

    llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
    template = """
    You are a world class journalist & researcher, you are extremely good at find most relevant articles to certain topic;
    {response_str}
    Above is the list of search results for the query {query}.
    Please choose the best 3 articles from the list, return ONLY an array of the urls, do not include anything else; return ONLY an array of the urls, do not include anything else
    """

    prompt_template = PromptTemplate(
        input_variables=["response_str", "query"], template=template
    )

    article_picker_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)

    urls = article_picker_chain.predict(response_str=response_str, query=query)

    # Convert string to list
    print("urls", urls)
    url_list = json.loads(urls)
    print(url_list)

    return url_list


# get content from each article & create a vector database


def get_content_from_urls(urls):
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    return data


def summarise(data, query):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=3000, chunk_overlap=200, length_function=len
    )
    text = text_splitter.split_documents(data)

    llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
    template = """
    {text}
    You are a world class journalist, and you will try to summarise the text above in order to create a twitter thread about {query}
    Please follow all of the following rules:
    1/ Make sure the content is engaging, informative with good data
    2/ Make sure the content is not too long, it should be no more than 3-5 tweets
    3/ The content should address the {query} topic very well
    4/ The content needs to be viral, and get at least 1000 likes
    5/ The content needs to be written in a way that is easy to read and understand
    6/ The content needs to give audience actionable advice & insights too
    7/ Limit Response to 1000 tokens

    SUMMARY:
    """

    prompt_template = PromptTemplate(
        input_variables=["text", "query"], template=template
    )

    summariser_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)

    summaries = []

    for chunk in enumerate(text):
        summary = summariser_chain.predict(text=chunk, query=query)
        summaries.append(summary)

    print(summaries)
    return summaries


def generate_thread(summaries, query):
    summaries_str = str(summaries)

    llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
    template = """
    {summaries_str}

    You are a world class journalist & twitter influencer, text above is some context about {query}
    Please write a viral twitter thread about {query} using the text above, and following all rules below:
    1/ The thread needs to be engaging, informative with good data
    2/ The thread needs to be around than 4 tweets
    3/ The thread needs to address the {query} topic very well
    4/ The thread needs to be viral, and get at least 1000 likes
    5/ The thread needs to be written in a way that is easy to read and understand
    6/ The thread needs to give audience actionable advice & insights too
    7/ Limit Response to 1000 tokens
    8/ Limit thread to be 280 characters or less
    9/ Use emojis to make the thread more engaging, but not too many

    TWITTER THREAD:
    """

    prompt_template = PromptTemplate(
        input_variables=["summaries_str", "query"], template=template
    )
    twitter_thread_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)

    twitter_thread = twitter_thread_chain.predict(
        summaries_str=summaries_str, query=query
    )

    return twitter_thread


def main():
    st.set_page_config(page_title="Free Tweet Generator", page_icon=":bird:")

    st.header("Web Tweeter :bird:")
    openApiKey = st.text_input("Please enter your OpenAI API Key")
    SerpApiKey = st.text_input("Please enter your SerpAPI API Key")
    query = st.text_input("Topic of twitter thread")
    # make a submit button
    submit = st.button("Generate Thread")

    if submit:
        openai.api_key = openApiKey
        st.write("Generating twitter thread for: ", query)
        search_results = search(query, SerpApiKey)
        with st.expander("search results"):
            st.info(search_results)
        urls = find_best_article_urls(search_results, query)
        with st.expander("best urls"):
            st.info(urls)
        data = get_content_from_urls(urls)
        with st.expander("data"):
            st.info(data)
        summaries = summarise(data, query)
        with st.expander("summaries"):
            st.info(summaries)
        thread = generate_thread(summaries, query)
        with st.expander("Tweets"):
            st.info(thread)


if __name__ == "__main__":
    main()
