import datetime

import dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import ChatMessage
from langchain.tools import tool
import streamlit as st

from tools.PolygonAPIClient import PolygonClient

dotenv.load_dotenv()
config = dotenv.dotenv_values()


@tool("get_ticker_news")
def get_ticker_news(query):
    """
    Get ticker news for a given symbol.
    """

    current_date_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    llm = ChatOpenAI(openai_api_key=config["OPENAI_API_KEY"],
                     streaming=False,
                     model_name="gpt-4",
                     temperature="0.5",
                     max_tokens="2048")

    with st.spinner("Internal Agent [Ticker News Agent] is transforming your request..."):
        json_request_url = llm(
            [ChatMessage(role="user", content=f"""
                                    The user asked the following query to another GPT agent:
    
                                    - {query}
    
                                    - Here is the current date in case you might need it: {current_date_string}
    
                                    ---
    
                                    Based on the user's query, you need to query an API to provide the required stock news data to the
                                    other agent. The agent might need this information to make a decision about a stock, or something
                                    else. Still, your only task is to create the API request parameters for the other agent.
                                    Your task is to generate the API request parameters with a "space character" between each parameter.
    
                                    The API request parameters are:
                                    - ticker_symbol : The symbol of the ticker in the financial / stocks market (e.g. AAPL)
                                    - max_limit : The maximum number of news to get. (e.g. 5)
                                                Please not that the maximum limit is 10, and further value will
                                                still return 10 news.
                                    - start_date : The start date of the news to get. (e.g. 2023-01-09)
                                    - end_date : The end date of the news to get. (e.g. 2023-01-09)
    
                                    ---
    
                                    Here is an example of what you must return:
    
                                    AAPL 5 2023-01-09 2023-01-09
    
                                    ---
                                """)]
        )
    st.success("Internal Agent [Ticker News Agent] has transformed your request successfully!")

    with st.spinner("Internal Agent [Ticker News Agent] is querying the API..."):
        parameters = json_request_url.content.split(" ")
        ticker_symbol = parameters[0].strip()
        max_limit = parameters[1].strip()
        start_date = parameters[2].strip()
        end_date = parameters[3].strip()

        p = PolygonClient()
        news = p.get_news(
            ticker_symbol=ticker_symbol,
            max_limit=max_limit,
            start_date=start_date,
            end_date=end_date
        )
    st.success("Internal Agent [Ticker News Agent] has queried the API successfully!")

    with st.expander("Reference Data [News API]", expanded=False):
        st.warning("\n\n" + str(news) + "\n\n")
        st.warning("Source: \n\n [1]  " + config["POLYGON_API_URL"] + "\n\n")

    return news


if __name__ == "__main__":
    print(get_ticker_news("What is the news for AAPL?"))
