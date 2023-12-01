import os
import pandas as pd
import evadb
import psycopg2
import openai


db_params = {
    'user': 'postgres',
    'password': 'xxclassifiedxx',
    'host': 'localhost',
    'port': '5432',
    'database': 'postgres'
}

stocks = ['AAPL', 'AMD', 'AMZN', 'GOOGL', 'INTC', 'JPM', 'MA', 'META', 'MSFT', 'NVDA', 'TSLA', 'V']

def reset_all_postgres():
    drop_table = "DROP TABLE IF EXISTS stock_data;"
    create_table = """
    CREATE TABLE stock_data (
        stock_symbol varchar,
        date varchar,
        open numeric,
        close numeric,
        high numeric,
        low numeric,
        volume bigint
    );
    """

    connection = psycopg2.connect(**db_params)
    cursor = connection.cursor()

    cursor.execute(drop_table)
    cursor.execute(create_table)
    connection.commit()

    cursor.close()
    connection.close()

def reset_all_eva():
    drop_table = """
        USE postgres_stock_data {
            DROP TABLE IF EXISTS stock_table
        }
    """

    drop_db = "DROP DATABASE IF EXISTS postgres_stock_data;"

    cursor = evadb.connect().cursor()
    cursor.query(drop_table).df()
    cursor.query(drop_db).df()
    cursor.close()

def setup_database():
    conditional_drop = "DROP DATABASE IF EXISTS postgres_stock_data;"
    connect_postgres = f"CREATE DATABASE postgres_stock_data WITH ENGINE = 'postgres', PARAMETERS = {db_params};"

    cursor = evadb.connect().cursor()

    cursor.query(conditional_drop)
    cursor.query(connect_postgres).df()

def setup_tables():
    drop_table = """
        USE postgres_stock_data {
            DROP TABLE IF EXISTS stock_table
        }
    """

    create_stock_table = """
    USE postgres_stock_data {
        CREATE TABLE IF NOT EXISTS stock_data (stock_symbol TEXT, date DATE, open NUMERIC, close NUMERIC, high NUMERIC, low NUMERIC, volume BIGINT)
    }
    """
    
    cursor = evadb.connect().cursor()
    cursor.query(drop_table).df()
    cursor.query(create_stock_table).df()
    cursor.close()

    upload_stock_data(merge_stock_data())

def recent_price_query(ticker: str, num_days: str):
    cursor = evadb.connect().cursor()

    query = f"""
    USE postgres_stock_data {{
        SELECT stock_symbol, date, close
        FROM stock_data
        WHERE stock_symbol = '{ticker}'
        ORDER BY date DESC
        LIMIT {num_days}
    }}
    """

    df = cursor.query(query).df()
    cursor.close()

    return df

def forecast_price(ticker: str, horizon: str, indicator: str = 'close'):
    train_forecast_func(ticker, indicator)

    cursor = evadb.connect().cursor()

    df = cursor.query(f"SELECT Forecast{ticker}({horizon}) ORDER BY DATE DESC;").df()

    return df

def train_forecast_func(ticker: str, indicator: str = 'close'):
    cursor = evadb.connect().cursor()

    conditional_drop = f"DROP FUNCTION IF EXISTS Forecast{ticker}"

    cursor.query(conditional_drop).df()
    cursor.query(f"""
        CREATE FUNCTION Forecast{ticker} FROM
            (
                SELECT stock_symbol, date, {indicator}
                FROM postgres_stock_data.stock_data
                WHERE stock_symbol = '{ticker}'
            )
        TYPE Forecasting
        PREDICT '{indicator}'
        TIME 'date'
        ID 'stock_symbol'
        MODEL 'AutoTheta'
        FREQUENCY 'D'
    """).df()

    cursor.close()

def setup_ai_funcs():
    cursor = evadb.connect().cursor()
    
    conditional_drop = "DROP FUNCTION IF EXISTS StockPriceForecast;"

    cursor.query(conditional_drop).df()
    cursor.query("""
        CREATE FUNCTION StockPriceForecast FROM
            (
                SELECT stock_symbol, date, close
                FROM postgres_stock_data.stock_data
                WHERE stock_symbol = 'GOOG'
            )
        TYPE Forecasting
        PREDICT 'close'
        TIME 'date'
        ID 'stock_symbol'
        MODEL 'AutoTheta'
        FREQUENCY 'D'
    """).df()

    cursor.close()

def generate_analysis_summary(ticker: str):
    cursor = evadb.connect().cursor()

    df = cursor.query(f"USE postgres_stock_data {{SELECT * FROM stock_data WHERE stock_symbol = '{ticker}' ORDER BY date DESC LIMIT 10}}").df()

    selected_columns = ['stock_symbol', 'date', 'open', 'close', 'high', 'low', 'volume']
    filtered_data = df[selected_columns].copy()

    # Converting data to a dictionary
    data = filtered_data.to_dict(orient='list')

    stock_data = pd.DataFrame(data)

    # Generating a textual summary of the stock data
    summary = f"Summary of {ticker} stock data:\n"
    summary += f"Date Range: {stock_data['date'].min()} to {stock_data['date'].max()}\n"
    summary += f"Opening Price Range: ${stock_data['open'].min()} to ${stock_data['open'].max()}\n"
    summary += f"Closing Price Range: ${stock_data['close'].min()} to ${stock_data['close'].max()}\n"
    summary += f"Highest Price Recorded: ${stock_data['high'].max()}\n"
    summary += f"Lowest Price Recorded: ${stock_data['low'].min()}\n"
    summary += f"Total Volume Traded: {stock_data['volume'].sum()} shares\n"

    return summary

def create_digest():
    client = openai.OpenAI(
    api_key='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',
    )

    messages = [ {"role": "system", "content":  
              "You are an intelligent assistant."} ]

    for ticker in stocks:
        summary = generate_analysis_summary(ticker)
        messages.append( 
            {"role": "user", "content": summary}, 
        )

    messages.append(
        {"role": "user", "content": "Create a news digest of all the provided stocks, and provide only a high level overview"}
    )
    chat = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=messages 
    )
    reply = chat.choices[0].message.content 

    messages.append({"role": "assistant", "content": reply})

    return reply

def create_analysis_report(ticker: str):
    client = openai.OpenAI(
    api_key='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',
    )

    messages = [ {"role": "system", "content":  
              "You are an intelligent assistant."} ]

    summary = generate_analysis_summary(ticker)

    messages.append( 
        {"role": "user", "content": summary}, 
    )
    messages.append(
        {"role": "user", "content": "Generate an analysis report on the stock prices provided for the stock"}
    )
    chat = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=messages 
    )
    reply = chat.choices[0].message.content 

    messages.append({"role": "assistant", "content": reply})

    return reply

def create_comparison_report(ticker1: str, ticker2: str):
    client = openai.OpenAI(
    api_key='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',
    )

    messages = [ {"role": "system", "content":  
              "You are an intelligent assistant."} ]

    for ticker in [ticker1, ticker2]:
        summary = generate_analysis_summary(ticker)
        messages.append( 
            {"role": "user", "content": summary}, 
        )
    
    messages.append(
        {"role": "user", "content": "Create a comparison report between the two stocks and their provided stock data"}
    )
    chat = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=messages 
    )
    reply = chat.choices[0].message.content 
    # print(f"ChatGPT: {reply}") 
    messages.append({"role": "assistant", "content": reply})

    return reply

# merges multiple stock .csv files into one
def merge_stock_data():
    merged_data = pd.DataFrame()

    stock_folder_path = 'stonkprices'

    for ticker in stocks:
        stock_data_filepath = os.path.join(stock_folder_path, f'{ticker}.csv')
        stock_data = pd.read_csv(stock_data_filepath)

        stock_data.columns = stock_data.columns.str.lower() # convert column names to lower-case
        stock_data['stock_symbol'] = ticker # add ticker symbol attribute

        merged_data = merged_data._append(stock_data)
    
    return merged_data

# upload merged data to evadb
def upload_stock_data(merged_data: pd.DataFrame):
    # SQL command to insert fresh data
    cursor = evadb.connect().cursor()


    # # Insert fresh data
    # cursor.copy_expert(sql=insert_data_sql, file=merged_data.to_csv(index=False, sep=','))
    # connection.commit()

    # iteratively insert data row by row
    # for idx, row in merged_data.iterrows():
    #     insert_row = f"""
    #     USE postgres_stock_data {{
    #     INSERT INTO stock_data VALUES ('{row['stock_symbol']}', '{row['date']}', '{row['open']}', '{row['close']}', '{row['high']}', '{row['low']}', '{row['volume']}')
    #     }}
    #     """
    #     cursor.query(insert_row).df()

    # cursor.close()

    # Updated insert method
    if 'adj close' in merged_data.columns:
        merged_data.drop(columns='adj close', inplace=True)

    rows = [tuple(row) for row in merged_data.values]

    batch_size = 5000
    # Batch insert data into the database
    for i in range(0, len(rows), batch_size):
        # Get a batch of rows
        batch = rows[i:i + batch_size]

        # Create the string representation of the batch
        insert_values = ", ".join(["('" + "', '".join(map(str, row)) + "')" for row in batch])

        # Create the full SQL INSERT query for this batch
        insert_query = f"""
        USE postgres_stock_data {{
            INSERT INTO stock_data (date, open, high, low, close, volume, stock_symbol)
            VALUES {insert_values}
        }}
        """

        # Execute the INSERT query
        cursor.query(insert_query).df()

    cursor.close()

# if __name__ == '__main__':
#     reset_all_eva()

#     setup_database()
#     setup_tables()
#     setup_ai_funcs()
#     print("donejamin")
#     # init_price_table()
