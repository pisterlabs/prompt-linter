import datetime

import dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import ChatMessage
from langchain.tools import tool
import streamlit as st

from tools.PolygonAPIClient import PolygonClient


dotenv.load_dotenv()
config = dotenv.dotenv_values()


@tool("get_technical_data", return_direct=True)
def get_technical_data(query):
    """
    Get technical data for a given symbol.
    """

    current_date_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    llm = ChatOpenAI(openai_api_key=config["OPENAI_API_KEY"],
                     streaming=False,
                     model_name="gpt-4",
                     temperature="0.5",
                     max_tokens="2048")

    with st.spinner("Internal Agent [Technical Data Agent] is transforming your request..."):
        json_request_url = llm(
            [ChatMessage(role="user", content=f"""
                    The user asked the following query to another GPT agent:
    
                    - {query}
                    
                    - Here is the current date in case you might need it: {current_date_string}
    
                    ---
    
                    Based on the user's query, you need to query an API to provide the required technical data to the
                    other agent. The agent might need this information to make a decision about a stock, or something
                    else. Still, your only task is to create the API request parameters for the other agent.
                    Your task is to generate the API request parameters with a "space character" between each parameter.
    
                    The API request parameters are:
                    - ticker_symbol: The symbol of the ticker in the financial / stocks market (e.g. AAPL)
                    - time_window: The period of time to get the technical data for, the options are:
                          - second, minute, hour, day, week, month, quarter, year
                    - start_date: The start date of the time window to get the technical data for. (e.g. 2023-01-09)
                    - end_date: The end date of the time window to get the technical data for. (e.g. 2023-01-09)
                    - is_adjusted: Whether the technical data are adjusted for stock splits or not. (e.g. true)
                    - max_limit: The maximum number of technical data to get. (e.g. 10) Please not that the maximum limit
                     is 32, and further requests will still return 32 technical data.
    
                    ---
    
                    Here is an example of what you must return:
    
                    AAPL day 2023-01-09 2023-01-09 true 10
    
                    ---
                """)]
        )
    st.success("Internal Agent [Technical Data Agent] has transformed your request successfully!")

    with st.spinner("Internal Agent [Technical Data Agent] is querying the API..."):
        parameters = json_request_url.content.split(" ")
        ticker_symbol = parameters[0].strip()
        time_window = parameters[1].strip()
        start_date = parameters[2].strip()
        end_date = parameters[3].strip()
        is_adjusted = parameters[4].strip()
        max_limit = parameters[5].strip()

        p = PolygonClient()
        tickers = p.get_tickers(
            ticker_symbol=ticker_symbol,
            time_window=time_window,
            start_date=start_date,
            end_date=end_date,
            is_adjusted=is_adjusted,
            max_limit=max_limit)
    st.success("Internal Agent [Technical Data Agent] has queried the API successfully!")

    with st.expander("Reference Data [Technical Data API]", expanded=False):
        st.warning("\n\n" + str(tickers) + "\n\n")
        st.warning("Source: \n\n [1]  " + config["POLYGON_API_URL"] + "\n\n")

    return tickers


if __name__ == "__main__":
    print(get_technical_data("What are the last 30 days of prices for AAPL?"))
