import pandas as pd
from pandasai import PandasAI
from chatbot_settings import ChatBotSettings

chatbotSettings = ChatBotSettings()


# Sample DataFrame
df = pd.DataFrame({
    "country": ["United States", "United Kingdom", "France", "Germany", "Italy", "Spain", "Canada", "Australia", "Japan", "China"],
    "gdp": [19294482071552, 2891615567872, 2411255037952, 3435817336832, 1745433788416, 1181205135360, 1607402389504, 1490967855104, 4380756541440, 14631844184064],
    "happiness_index": [6.94, 7.16, 6.66, 7.07, 6.38, 6.4, 7.23, 7.22, 5.87, 5.12]
})


# Assuming you have the data in a DataFrame named 'df'
df = pd.DataFrame({
    'name': ['John Doe','John Doe','John Doe'],
    'date': ['2023-04-13','2023-03-13','2023-02-13'],
    'How are you doing overall?': ['Good','Good','Good'],
    'Does anyting hurt?': ['No','No','Yes'],
    'What is your level of pain? (1-5)': [4,5,2],
    'Do you have any other symptoms?': ['Legs','Legs','Legs']
    # "name": ["United States", "United Kingdom", "France"],
    # "gdp": [19294482071552, 2891615567872, 2411255037952],
    # "happiness_index": [6.94, 7.16, 6.66]
})

# Instantiate a LLM
from pandasai.llm.openai import OpenAI
llm = OpenAI()

while True:
    pandas_ai = PandasAI(llm, conversational=False)
    i = input()
    response = pandas_ai.run(df, prompt=i)

    print(response)