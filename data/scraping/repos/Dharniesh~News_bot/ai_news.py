import streamlit as st
from gnews import GNews
import pandas as pd
import time
import openai
import requests
from bs4 import BeautifulSoup

openai.api_key=st.secrets['openai_api']

def ask_GPT(news):
    prompt = f"""
    Your objective is to create a summary of a news webpage given by presenting\
    the most crucial information in bullet points.
    
    Summarize the News below, delimited by triple
    backticks.
    
    News: ```{news}```
    """
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        temperature=0,
        max_tokens=4096,
        messages=[
            {"role": "user", "content": prompt }
        ]
    )
    return completion.choices[0].message.content

st.markdown("<div style='text-align: center;'><h1>News Search App</h1></div>", unsafe_allow_html=True)

note_style = "<div style='text-align: center; color: #3366ff; font-size: 18px;'>Note: You can either enter the period or start and end dates.</div>"
st.markdown(note_style, unsafe_allow_html=True)
st.write("\n\n")

col1, col2 = st.columns(2)

with col1:
    period_days = st.text_input("Days Before News Request (Leave blank to skip):")
    
    st.markdown("<div style='text-align: center; font-weight: bold;'>OR</div>", unsafe_allow_html=True)
    
    start_date_disabled = False
    end_date_disabled = False
    
    if period_days:
        start_date_disabled = True
        end_date_disabled = True
    
    start_date = st.date_input("Enter start date (Leave blank to skip):", key="start_date", disabled=start_date_disabled, value=None)
    if start_date_disabled:
        start_date = None
    
    end_date = st.date_input("Enter end date *:", key="end_date", disabled=end_date_disabled, value=None)
    if end_date_disabled:
        end_date = None

with col2:
    exclude_websites = st.text_input("Enter websites URL to exclude  (Leave blank to skip):")
    max_results = st.text_input("News Count (default = 1):")
    
    search_button_clicked = st.button("Search")

if search_button_clicked:
    max_results = int(max_results) if max_results else 1
    period = f"{period_days}d" if period_days else None

    formatted_start_date = (start_date.year, start_date.month, start_date.day) if start_date else None
    formatted_end_date = (end_date.year, end_date.month, end_date.day) if end_date else None

    parameters = {
        "language": "en",
        "country": "GB",
        "max_results": max_results,
        "period": period,
        "start_date": formatted_start_date,
        "end_date": formatted_end_date,
        "exclude_websites": exclude_websites
    }

    google_news = GNews(**parameters)
    if not max_results or int(max_results) > 5:
        st.warning("Please enter a value less than or equal to 5 for max_results.")
    else:
        news_list = ["what is generative ai", "ai script generator", "creating ai", "ai title generator"]
        results = []
        for j in news_list:
            results_k = google_news.get_news(j)
            results.extend(results_k)
        
        publisher_titles = []
        publisher_hrefs = []

        for result in results:
            publisher_info = result.get('publisher', {})
            publisher_titles.append(publisher_info.get('title', ''))
            publisher_hrefs.append(publisher_info.get('href', ''))

        df = pd.DataFrame(results)

        if search_button_clicked and df is not None:
            st.write("Search Results:")
            st.dataframe(df)

        cgpt_text = []
        txt_summ = []

        with open('summaries.txt', 'w', encoding='utf-8') as file:
            for i, URL in enumerate(df['url']):
                try:
                    r = requests.get(URL)
                    r.raise_for_status()
                    soup = BeautifulSoup(r.text, 'html.parser')
                    results = soup.find_all(['h1', 'p'])
                    text = [result.get_text() for result in results]
                    news_article = ' '.join(text)
                    cgpt_text.append(news_article)
                    summary_txt = ask_GPT(news_article)
                    txt_summ.append(summary_txt)
                    df.loc[i, 'summary_title'] = df.loc[i, 'title']

                    file.write(f"Title: {df['title'][i]}\n")
                    file.write(f"URL: {URL}\n")
                    file.write(f"Summary: {summary_txt}\n\n")
                    
                    print(news_article)
                    print('------------------------------------------------------------')

                    time.sleep(1)

                except requests.exceptions.HTTPError as e:
                    if e.response.status_code != 200:
                        file.write(f"Failed URL: {URL}\n\n")
                    else:
                        print(f"Error processing URL: {URL}\nError message: {e}")

                    time.sleep(1)

        with open('summaries.txt', 'r', encoding='utf-8') as file:
            file_contents = file.read()

        st.download_button(
            label="Download Summaries",
            data=file_contents,
            file_name='summaries.txt',
            mime='text/plain',
        )
