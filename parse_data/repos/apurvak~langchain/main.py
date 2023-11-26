# This is a sample Python script.
from langchain.callbacks import StdOutCallbackHandler
from langchain.chains import SQLDatabaseSequentialChain
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from langchain.llms import NLPCloud
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain, PromptTemplate

import SqlQuotronChain


def process_document(question = 'what is the highest revenue?'):


    db = SQLDatabase.from_uri("postgresql://superset:superset@localhost:5432/superset", ignore_tables= ['ab_permission_view_role', 'druiddatasource_user', 'clusters', 'alembic_version', 'ab_permission_view', 'sl_dataset_columns', 'sl_columns', 'sl_dataset_users', 'user_attribute', 'ab_role', 'train', 'column_histories', 'embedded_dashboards', 'sql_metrics', 'alert_logs', 'sl_table_columns', 'keyvalue', 'columns', 'tab_state', 'sl_tables', 'quotron_feedback', 'rls_filter_roles', 'row_level_security_filters', 'cache_keys', 'saved_query', 'report_schedule', 'birth_names', 'ab_permission', 'favstar', 'table_schema', 'dashboard_roles', 'ab_register_user', 'logs', 'slices', 'dashboards', 'conversations', 'tables', 'access_request', 'filter_sets', 'metrics', 'annotation_layer', 'key_value', 'report_schedule_user', 'rls_filter_tables', 'sqlatable_user', 'tagged_object', 'query', 'ab_user_role', 'dashboard_slices', 'dashboard_email_schedules', 'annotation', 'alerts', 'url', 'report_recipient', 'dynamic_plugin', 'report_execution_log', 'alert_owner', 'sl_dataset_tables', 'sql_observations', 'table_columns', 'tag', 'ab_user', 'ab_view_menu', 'sl_datasets', 'datasources', 'css_templates', 'slice_email_schedules', 'dbs', 'slice_user', 'dashboard_user'])
    #
    _DEFAULT_TEMPLATE = """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
    Use the following format:

    Question: "Question here"
    Query: "SQL Query to run"
    Result: "Result of the SQLQuery"
    Answer: "Final answer here"

    Only use the following tables:

    cleaned_sales_data

    If someone asks for the table foobar, they really mean the employee table.

    Question: {input}"""
    PROMPT = PromptTemplate(
        input_variables=["input", "dialect"], template=_DEFAULT_TEMPLATE
    )
    prompt = PROMPT
    # llm = OpenAI(temperature=0, verbose=True)
    llm = NLPCloud(verbose=True)
    db_chain = SqlQuotronChain.SQLDatabaseSequentialChain.from_llm(llm=llm, database= db, verbose=True)

    answer = db_chain.run(query ="What is quarterly sales for each product line?")
    print(answer)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    process_document()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
