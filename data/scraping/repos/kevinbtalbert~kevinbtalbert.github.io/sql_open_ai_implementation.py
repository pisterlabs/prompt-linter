#shell: !pip install openai
import openai
import cml.data_v1 as cmldata

openai.api_key = "sh-xxxx"

def get_cdw_table_schema(database, table_name):
    CONNECTION_NAME = "default-impala"
    conn = cmldata.get_connection(CONNECTION_NAME)
    SQL_QUERY = "DESCIBE " + "`" + str(database) + "`." + str(table_name) + "; "
    dataframe = conn.get_pandas_dataframe(SQL_QUERY)
    return dataframe.to_string()

def run_sql(sql):
    dataframe = conn.get_pandas_dataframe(sql)
    return dataframe.to_string()

def get_response(question, context, engine):

    enhanced_question = """Build a SQL query to solve the given question based on given table structure. """ + question
    
    response = openai.ChatCompletion.create(
        model=engine,
        messages=[
            {"role": "system", "content": str(context)},
            {"role": "user", "content": str(enhanced_question)}
            ]
    )
    
    return response['choices'][0]['message']['content']

def main():
    question = "How many people are attending university X?"
    context = get_cdw_table_schema("database-name-here", "table-name-here")
    engine_options = ['gpt-3.5-turbo', 'gpt-4']
    
    ## Response with GPT 3.5
    res1 = get_response(question, context, engine_options[0])
    print(res1)
    ## You may have to strip out some text the LLM provides back to get the raw SQL but ideally...
    print(run_sql(res1))

    ## Response with GPT 4
    res2 = get_response(question, context, engine_options[1])
    print(run_sql(res2))
    

if __name__ == "__main__":
    main()

