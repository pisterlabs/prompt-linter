'''
PRELIMINARY NOTES:
Upgrade version of PandasAI.py
pip install --upgrade pandas pandasai
'''


import pandas as pd
from pandasai import SmartDataframe

# Sample DataFrame
df = pd.DataFrame({
    "country": ["United States", "United Kingdom", "France", "Germany", "Italy", "Spain", "Canada", "Australia", "Japan", "China"],
    "gdp": [19294482071552, 2891615567872, 2411255037952, 3435817336832, 1745433788416, 1181205135360, 1607402389504, 1490967855104, 4380756541440, 14631844184064],
    "happiness_index": [6.94, 7.16, 6.66, 7.07, 6.38, 6.4, 7.23, 7.22, 5.87, 5.12]
})

# Instantiate a LLM
from pandasai.llm import OpenAI
llm = OpenAI(api_token="generators-openai-token")

df = SmartDataframe(df, config={"llm": llm})
#print(df.chat('Which are the 5 happiest countries?'))

#print(df.chat("plot gdp vs happiness_index"))

df.plot_histogram("gdp")  #saved in exports>charts