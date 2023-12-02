import datetime

import dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import ChatMessage
from langchain.tools import tool
import streamlit as st

from tools.PolygonAPIClient import PolygonClient


dotenv.load_dotenv()
config = dotenv.dotenv_values()


@tool("get_dividends", return_direct=True)
def get_dividends(query):
    """
    Get dividends for a given symbol.
    """

    current_date_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    llm = ChatOpenAI(openai_api_key=config["OPENAI_API_KEY"],
                     streaming=False,
                     model_name="gpt-4",
                     temperature="0.5",
                     max_tokens="2048")

    with st.spinner("Internal Agent [Dividends Agent] is transforming your request..."):
        json_request_url = llm(
            [ChatMessage(role="user", content=f"""
                                The user asked the following query to another GPT agent:
    
                                - {query}
    
                                - Here is the current date in case you might need it: {current_date_string}
    
                                ---
    
                                Based on the user's query, you need to query an API to provide the required dividends data to the
                                other agent. The agent might need this information to make a decision about a stock, or something
                                else. Still, your only task is to create the API request parameters for the other agent.
                                Your task is to generate the API request parameters with a "space character" between each parameter.
    
                                The API request parameters are:
                                - ticker_symbol : The symbol of the ticker in the financial / stocks market (e.g. AAPL)
                                - frequency : The frequency of the dividends, the options are:
                                        - 0: One Time
                                        - 1: Annual
                                        - 2: Biannual
                                        - 4: Quarterly
                                        - 12: Monthly      
                                - dividend_type : The type of the dividends, the options are:
                                        - CD: Consistent Dividend
                                        - SC: Cash Dividend
                                        - LT: Long Term Dividend
                                        - ST: Short Term Dividend
                                - max_limit : The maximum number of dividends to get. (e.g. 5)
                                            Please not that the maximum limit is 10, and further value will
                                            still return 10 dividends.
    
                                ---
    
                                Here is an example of what you must return:
    
                                AAPL 4 CD 5
    
                                ---
                            """)]
        )
    st.success("Internal Agent [Dividends Agent] has transformed your request successfully.")

    with st.spinner("Internal Agent [Dividends Agent] is querying the API..."):
        parameters = json_request_url.content.split(" ")
        ticker_symbol = parameters[0].strip()
        frequency = parameters[1].strip()
        dividend_type = parameters[2].strip()
        max_limit = parameters[3].strip()

        p = PolygonClient()
        dividends_data = p.get_dividends(
            ticker_symbol=ticker_symbol,
            frequency=frequency,
            dividend_type=dividend_type,
            max_limit=max_limit
        )
    st.success("Internal Agent [Dividends Agent] has queried the API successfully.")

    with st.expander("Reference Data [Dividends API]", expanded=False):
        st.warning("\n\n" + str(dividends_data) + "\n\n")
        st.warning("Source: \n\n [1]  ", config["POLYGON_API_URL"] + "\n\n")

    return dividends_data


if __name__ == "__main__":
    query = "What is the dividend for MSFT?"
    print(get_dividends(query))
