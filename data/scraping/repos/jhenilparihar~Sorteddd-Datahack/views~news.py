import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
import openai


new = ""


openai.api_key = "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"


def gpt3_completion(prompt, engine='text-davinci-002', temp=0.7, top_p=1.0, tokens=400, freq_pen=0.0, pres_pen=0.0, stop=['<<END>>']):
    prompt = prompt.encode(encoding='ASCII', errors='ignore').decode()
    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        temperature=temp,
        max_tokens=tokens,
        top_p=top_p,
        frequency_penalty=freq_pen,
        presence_penalty=pres_pen,
        stop=stop)
    text = response['choices'][0]['text'].strip()
    return text


news = [
    {
        "title": "Adani shares slip, Group m-cap falls by Rs 49,400 cr",
        "url": "https://economictimes.indiatimes.com/markets/stocks/news/adani-shares-slip-group-m-cap-falls-by-rs-49400-cr/articleshow/99074180.cms",
        "date": "March 29, 2023",
        "status": "Negative"
    },
    {
        "title": "Adani Repays $500 Million Bridge Loan to Regain Investor Faith",
        "url": "https://www.bloomberg.com/news/articles/2023-03-08/adani-repays-500-million-bridge-loan-to-regain-investor-faith",
        "date": "March 8, 2023",
        "status": "Positive"
    },
    {
        "title": "Adani Enterprises stock exits NSE’s additional security framework after a month",
        "url": "https://www.financialexpress.com/market/adani-enterprises-exits-nses-additional-security-framework-after-a-month/3001534/",
        "date": "March 7, 2023",
        "status": "Negative"
    }
]


def load_view():
    col, __ = st.columns((10, 1))

    with col:
        st.subheader("How did news affect Adani's stock price")
        st.markdown("#")

    col, _ = st.columns((10, 1))

    for j, i in enumerate(new):
        prompt_ = f"Give me in answer word [positive/negative] if the following news in positive or negative: {i['title']}"
        response_ = gpt3_completion(prompt_)
        news[j]['status'] = response_.split(' ')[-1].capitalize()

    for i in news:

        with col:
            st.header(f"{i['title']}")
            st.markdown(
                f"[Check full story]({i['url']})")
            st.markdown(f"<h5>Date: {i['date']}</h5>", unsafe_allow_html=True)
            st.markdown(f"<h4>{i['status']}</h4>", unsafe_allow_html=True)
            df = px.data.gapminder()

            fig = px.scatter(
                df.query("year==2007"),
                x="gdpPercap",
                y="lifeExp",
                size="pop",
                color="continent",
                hover_name="country",
                log_x=True,
                size_max=60,
            )
            df2 = px.data.iris()
            fig2 = px.scatter(
                df2,
                x="sepal_width",
                y="sepal_length",
                color="sepal_length",
                color_continuous_scale="reds",
            )

            tab1, tab2, tab3 = st.tabs(
                ["Adani Enterprise", "Adani Power", "Adani Ports"])
            with tab1:
                chart_data = pd.DataFrame(
                    np.random.randn(20, 1),
                    columns=['Adani Enterprise'])
                st.line_chart(chart_data)
            with tab2:
                chart_data = pd.DataFrame(
                    np.random.randn(20, 1),
                    columns=['Adani Power'])

                st.line_chart(chart_data)
            with tab3:
                chart_data = pd.DataFrame(
                    np.random.randn(20, 1),
                    columns=['Adani Ports'])

                st.line_chart(chart_data)
