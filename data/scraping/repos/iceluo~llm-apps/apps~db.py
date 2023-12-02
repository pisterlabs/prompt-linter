import json
import openai
import os
import pandas as pd
import streamlit as st
from lib import schema, utils
from sqlalchemy import create_engine


DATABASES = ['reservation', 'gpt']
PROMT = '''
You're playing a role of dbot. dbot is a very skillful, and creative database administrator. It will only give accurate, highly optimized, well-written SQLs based on users' descriptions of the problem. It could also use its own judgment to generate meaningful, valid mock data if user requires so. Here's the context (in SQLs) about the database users will ask:

```
{context}
```

dbot will strictly follow these rules:
- If users ask it a question that could be answered with current context of the database, it return a valid, accurate SQL that could answer the question. The SQL will be wrapped in a JSON object with key "sql". The JSON shall also contain a key "type" that indicates the type of the SQL (SELECT, INSERT, UPDATE, DELETE, etc).
- If users ask questions can't be answered with current context of the database, it will return a message that says it can't answer the question. The message will be wrapped in a JSON object with key "error".
- The response MUST be a JSON object with key "sql" or "error". Otherwise, it will be considered as an error.

Here's user's question: ```{question}```. Answer it as dbot.
'''


def app():
    openai.api_key = os.environ['OPENAI_KEY'] or Exception(
        'No OPENAI_KEY found in environment')
    st.set_page_config(page_title="Postgres Assistant", page_icon=":robot:")
    st.header("Database Assistant")

    if 'engine' not in st.session_state:
        st.session_state['engine'] = None

    if 'db_name' not in st.session_state:
        st.session_state['db_name'] = None

    if 'schemas' not in st.session_state:
        st.session_state['schemas'] = None

    if "history" not in st.session_state:
        st.session_state["history"] = []

    with st.sidebar.expander(label='Database Setting'):
        db_name = st.selectbox('Select your database', DATABASES)
        if st.session_state['engine'] is None or st.session_state['db_name'] != db_name:
            st.session_state['engine'] = create_engine(
                f'postgresql://postgres:postgres@localhost/{db_name}')

        if st.button('Regenerate Schema'):
            with st.session_state['engine'].connect() as con:
                schemas = schema.load_from_db(con)
                schema.save_to_file(db_name, schemas)
                st.session_state['schemas'] = schemas

            st.write(f'Database schema for {db_name} generated')
        else:
            if schema.exists(db_name):
                st.session_state['schemas'] = schema.load_from_file(db_name)

    if not st.session_state['schemas']:
        st.warning(
            f'Database schema for {db_name} not loaded. Please "Regenerate Schema" first')
        return

    st.success(
        f'Database schema for {db_name} loaded. You can ask questions now!')

    input = st.text_area('Question:', key='input')
    if st.button('Submit'):
        prompt = PROMT.format(
            context=st.session_state['schemas'], question=input)
        with st.spinner('Thinking super hard...'):
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
            )

            output = response['choices'][0]['message']['content']
            print(output)
            output = json.loads(output)

            answer = ''
            if 'error' in output:
                answer = output['error']
                st.error(answer)
                return

            if 'sql' in output:
                answer = output['sql']
                st.write('## Suggested SQL')
                st.code(utils.format_sql(answer), language='sql')

                with st.session_state['engine'].connect() as con:
                    try:
                        if output['type'] == 'SELECT':
                            st.write('## Query Result')
                            df = pd.read_sql_query(answer, con)
                            st.dataframe(df)
                        else:
                            con.execute(answer)
                            st.success('Query executed successfully')
                    except Exception as e:
                        st.write(e)

        st.session_state["history"].append({'q': input, 'a': answer})
        if len(st.session_state["history"]) > 5:
            st.session_state["history"].pop(0)

    with st.sidebar.expander(label='History'):
        if st.session_state["history"]:
            history = st.session_state["history"]
            for i in range(len(history)):
                c = st.container()
                c.write(history[i]["q"])
                c.write(history[i]["a"])


if __name__ == "__main__":
    app()
