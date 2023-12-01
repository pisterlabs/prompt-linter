import streamlit as st
import requests
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import os
from dotenv import load_dotenv
import streamlit.components.v1 as components

# Load the environment variables from the .env file
load_dotenv()

# Retrieve the keys
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Initialize Claude API with the loaded key
anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)

st.set_page_config(
    page_title="Algabay",
    layout="wide",
    page_icon="🧊",
    initial_sidebar_state="expanded",
)

def get_stock_news(stock_name):
    response = requests.get(f"https://newsapi.org/v2/everything?q={stock_name}+company&apiKey={NEWS_API_KEY}")
    return response.json()["articles"][:10]

def ask_claude(stock_info, query):
    prompt = f"{HUMAN_PROMPT} {stock_info} {query}{AI_PROMPT}"
    completion = anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=300,
        prompt=prompt,
    )
    return completion.completion

def fintech_app():

    st.title("Algabay AI")
    Trading_view_ticker_tape = """
    <!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container">
    <div class="tradingview-widget-container__widget"></div>
    <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js" async>
    {
    "symbols": [
        {
        "proName": "FOREXCOM:SPXUSD",
        "title": "S&P 500"
        },
        {
        "proName": "FOREXCOM:NSXUSD",
        "title": "US 100"
        },
        {
        "proName": "FX_IDC:EURUSD",
        "title": "EUR to USD"
        },
        {
        "proName": "BITSTAMP:BTCUSD",
        "title": "Bitcoin"
        },
        {
        "proName": "BITSTAMP:ETHUSD",
        "title": "Ethereum"
        }
    ],
    "showSymbolLogo": true,
    "colorTheme": "dark",
    "isTransparent": false,
    "displayMode": "adaptive",
    "locale": "en"
    }
    </script>
    </div>
    """
    components.html(Trading_view_ticker_tape, height=90)
    # Dictionary of famous Indian stock companies with their symbols
    stocks_with_symbols = {
        "Apple": "AAPL",
        "Reliance Industries": "RELIANCE",
        "Tata Consultancy Services (TCS)": "TCS",
        "HDFC Bank": "HDFCBANK",
        "Infosys": "INFY",
        "Hindustan Unilever": "HINDUNILVR",
        "ICICI Bank": "ICICIBANK",
        "Bharti Airtel": "BHARTIARTL",
        "Kotak Mahindra Bank": "KOTAKBANK",
        "Maruti Suzuki India": "MARUTI",
        "State Bank of India": "SBIN",
        "Housing Development Finance Corporation (HDFC)": "HDFC",
        "ITC Limited": "ITC",
        "Bajaj Finance": "BAJFINANCE",
        "Asian Paints": "ASIANPAINT",
        "Wipro": "WIPRO",
        "Axis Bank": "AXISBANK",
        "Larsen & Toubro (L&T)": "LT",
        "Nestle India": "NESTLEIND",
        "Mahindra & Mahindra": "M&M",
        "Sun Pharmaceutical Industries": "SUNPHARMA",
    }

    # Dropdown for stock selection
    selected_stock = st.selectbox(
        'Select a stock:',
        options=list(stocks_with_symbols.keys())
    )

    if selected_stock:
        st.session_state.selected_stock = selected_stock
        stock_symbol = stocks_with_symbols[selected_stock]  # Retrieve the symbol for the selected stock
        # Set stock_info here
        stock_info = f"Information about following company: {st.session_state.selected_stock}. Strictly adhere to relevancy of the company and keep the answer short and precise."
    else:
        stock_symbol = "NZDCAD"
        # Optionally set a default stock_info here
        stock_info = "No stock selected"


    st.sidebar.title("Ask Algabay AI")
    with st.sidebar:
        user_query = st.text_input(f"Type your question about {selected_stock}:")
        if st.button("ask"):
            if user_query:
                response = ask_claude(stock_info, user_query)
                st.write(response)


    tradingview_info_code = f"""
        <div class="tradingview-widget-container">
            <div class="tradingview-widget-container__widget"></div>
            <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-symbol-info.js" async>
            {{
            "symbol": "{stock_symbol}",
            "width": 1000,
            "locale": "in",
            "isTransparent": false,
            "colorTheme": "dark"
            }}
            </script>
        </div>
    """
    components.html(tradingview_info_code, height=200)

    tradingview_chart_code = f"""
        <div class="tradingview-widget-container">
            <div id="tradingview_chart_{stock_symbol}"></div>
            <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
            <script type="text/javascript">
            new TradingView.widget(
                {{
                    "container_id": "tradingview_chart_{stock_symbol}",
                    "symbol": "{stock_symbol}",
                    "interval": "D",
                    "width": "100%",
                    "height": "400",
                    istransparent: true,
                    "colorTheme": "dark"
                    // Additional chart widget options
                }}
            );
            </script>
        </div>
        """
    components.html(tradingview_chart_code, height=450)

    col1, col2 = st.columns(2)
    # Stock information to prepend to Claude's prompt
    stock_info = ""
    if st.session_state.selected_stock:  # Updated reference here
        stock_info = f"Information about following company: {st.session_state.selected_stock}. Strictly adhere to relevancy of the company and keep the answer short and precise."


    # Display stock news in the left column
    with col1:
        st.subheader("Latest News")
        if st.session_state.selected_stock:  # Updated reference here
            news_articles = get_stock_news(st.session_state.selected_stock)  # Updated reference here
        else:
            # Display generic news if no stock selected
            news_articles = get_stock_news("Nifty 50") + get_stock_news("Sensex")
        for article in news_articles:
            st.write(f"**{article['title']}**")
            st.write(article["description"])
            st.write(f"[Read more]({article['url']})")
            st.write("---")

    # AI Assistant Interaction in the right column
    with col2:
            tradingview_info = f"""
            <div class="tradingview-widget-container">
            <div class="tradingview-widget-container__widget"></div>
            <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-technical-analysis.js" async>
            {{
            "interval": "1m",
            "width": 425,
            "isTransparent": false,
            "height": 450,
            "symbol": "{stock_symbol}",
            "showIntervalTabs": true,
            "locale": "in",
            "colorTheme": "dark"
            }}
            </script>
            </div>
            """
            components.html(tradingview_info, height=450)

    
        

if __name__ == "__main__":
    fintech_app()
