from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
from ohlc_data_process import calculate_dmi_rsi_mfi, fetch_data, analyze_ichimoku, analyze_supertrend, calculate_supertrend
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain import FewShotPromptTemplate, PromptTemplate
import random

load_dotenv()

openai_api_key = os.getenv('T_OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = openai_api_key

openai = OpenAI(
    #model_name='text-davinci-003',
    model_name='text-davinci-003',
    temperature=0.7
)


# create our examples
trend_examples = [
{
    "query": f"Ichmouku trend,Strong Bullish,Detected price,110.34,Supertrend,uptrend,"
                f" cross for supertrend,True,Current Price,95.123"
                f" What is your advice for trading for this trend and price values?",
    "answer": "Our Ichmouku trend is Strong bullish that is good for next time step, we can say 110.34 our resistance \
        because the detected price from ichmouku is greater than our current price 95.123. Also we have \
        croos line and trend changing step for our supertrend indicator within 5 hour and the supertrend indicator momuntum is uptrend. \
            You can hold if you have this coin. Or you can buy this coin because we have trend changing for uptrend \
                and have a strong bullish momentum from ichmouku. you can follow this holding opeartion \
                    our detected resistance price to 110.34"
}, {
    "query": f"Ichmouku trend,Bearish,Detected price,90.34,Supertrend,downtrend,"
                f" cross for supertrend,No,Current Price,95.123"
                f" What is your advice for trading for this trend and price values?",
    "answer": "Our ichmouku trend is bearish not string enough, detected price is 90.34 and we can say this is our \
        supported price because current price is greater than our detected price. Also we have a downtrend for supertrend indicator. \
            we can say you can wait for buying because we dont see any cross in supertrend or bullish movement \
                from our trade engine."
}
]

trend_prefix = """Answer the question based on the context below.
You are the trading advisor. Also you are expert on Supertrend and Ichmouku Cloud indicators.

Context: supertrend has 2 level: uptrend and downtrend. Ichmouku has 4 level Bullish, strong Bullish,
Bearish, Strong Bearish. our detected price is coming from ichmouku indicator and we can use support or resistance price.
If the detected price is lower than the current Price this detected price is support price if detected price is greater than current price the detected price is resistance level.
Dont forget you can find which price is gretaer or lower than the other between detected price and current price.
you must a evaluation to customer current stiation from this indicator values. then you should give a trading advice to custormer.
Here are some examples:
"""

# create our examples
rsi_examples = [
{
    "query": f"RSI,70.34,MFI,59.02,DMP,31.3,"
                f" DMN,12.77,ADX,41.26"
                f" What is your advice for trading for those indicator values?",
    "answer": "The RSI indicator value being above 70 at this moment indicates an overbought zone has been entered."
                "The MFI value trending above the average confirms the flow of money. On the DMi side, "
                "the DMP (positive directional movement indicator) value is above "
                "the DMN (negative directional movement indicator) value, "
                "and ADX is strongly trending above 25 and at 40 levels, "
                "indicating a strong bull trend that has entered in a short period of time. "
                "When considering the flow of money and the overbought zone, it may be advisable "
                "to take some profits and waiting next market movements."
}, {
    "query": f"RSI,40.14, MFI,41, DMP,21.01,"
                f"DMN,23.67,ADX,20.76."
                f" What is your advice for trading for those indicator values?",
    "answer": "The RSI indicator value dropping around 40 indicates approaching the selling zone. "
                "The MFI index also dropping around 40 supports this. Although ADX suggests that there is no strong trend below 25, "
                "it can be observed that DMN is above DMP, creating selling pressure. "
                "My recommendation would be to wait for a better buying opportunity at this point."
}
]

# create a example template
example_template = """
User: {query}
AI: {answer}
"""

# create a prompt example from above template
example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

# now break our previous prompt into a prefix and suffix
# the prefix is our instructions
rsi_prefix = """Answer the question based on the context below.
You are the trading advisor. Also you are expert on RSI, MFI and DMP, DMI indicators.

Context: RSI indicator value range betwwen 0-100. RSI value 70 and above meaning that overbought area.
So you should sell your coin. also value 80 is a extreme overbought and you act cautiously.
RSI value 30 that meaning is overselling area. and value 20 is extreme overselling are.
if the RSI value are 20-30s you should bought this coin. 30-70 range waiting or
you can look other indicator results.
MFI indicator value range betwen 0-100. MFI value 80 and above meaning that overbought area.
So you should sell your coin. also value 90 is a extreme overbought and you should act cautiously.
MFI value 20 that meaning is overselling area. and value 10 is extreme overselling are.
if the MFI value are 10-20s you should bought this coin. 20-80 range waiting or
you can look other indicator results.
Else it shows overselling condition between 0-25.
DMI indicator is a collection of indicators including DMP, DMI, and ADX. The Plus Direction Indicator DMP and
Minus Direction Indicator DMI show the current price direction. When the DMP is above DMN,
the current price momentum is up. When the DMN is above DMP, the current price momentum is down.
ADX measures the strength of the trend, either up or down; a reading above 25 indicates a strong trend.
Here are some examples:
"""
# and the suffix our user input and output indicator
suffix = """
User: {query}
AI: """


class TradingAdvisor:
    
    @staticmethod
    def create_advice_prompt_template(examples, prefix):
        few_shot_prompt_template = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix=prefix,
            suffix=suffix,
            input_variables=["query"],
            example_separator="\n\n"
        )
        return few_shot_prompt_template

    @staticmethod
    def fetch_current_data(symbol: str):
        df = fetch_data(symbol, '1mo', '1h')
        indicator_data = calculate_dmi_rsi_mfi(df)
        return indicator_data
    
    @staticmethod
    def get_advice(symbol: str):
        df = TradingAdvisor.fetch_current_data(symbol)
        rsi_14 = df.RSI_14.iloc[-1]
        mfi_14 = df.MFI_14.iloc[-1]
        dmp_14 = df.DMP_14.iloc[-1]
        dmn_14 = df.DMN_14.iloc[-1]
        adx_14 = df.ADX_14.iloc[-1]
        
        trend_ich, base_price = analyze_ichimoku(df)
        spr_trend = calculate_supertrend(df)
        trend_super, cross_super = analyze_supertrend(spr_trend) 
        
        # print(trend_ich, base_price, trend_super, cross_super, df.iloc[-1]["Close"])
        
        rsi_query = f"RSI,{rsi_14:.2f},MFI,{mfi_14:.2f},DMP,{dmp_14:.2f}," \
                f" DMN,{dmn_14:.2f},ADX,{adx_14:.2f}"     
                
        trend_query = f"Ichmouku trend,{trend_ich},Detected price,{base_price[0]:.2f},Supertrend,{trend_super}," \
                f" cross for supertrend,{cross_super},Current Price,{df.iloc[-1]['Close']:.2f}"   

        rsi_prompt_template = TradingAdvisor.create_advice_prompt_template(rsi_examples, rsi_prefix)
        trend_prompt_template = TradingAdvisor.create_advice_prompt_template(trend_examples, trend_prefix)
        templates = [rsi_prompt_template, trend_prompt_template]
        queries = [rsi_query, trend_query]
        
        if cross_super:
            prompt_template = trend_prompt_template
            return openai(
                prompt_template.format(
                    query=trend_query
                )
            )
        else:
            rand = random.randint(0,1)
            prompt_template = templates[rand]
            return openai(
                    prompt_template.format(
                        query=queries[rand]
                    )
                )
        
if __name__ == '__main__':
    tradv = TradingAdvisor.get_advice("SOL-USD")
    print(tradv)
