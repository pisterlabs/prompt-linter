from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime
import pandas as pd
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 3, 16),
}


def execute_query_on_db(db_name, query):
    pg_hook = PostgresHook(postgres_conn_id=db_name)
    pg_conn = pg_hook.get_conn()
    pg_cursor = pg_conn.cursor()
    pg_cursor.execute(query)
    columns = [desc[0] for desc in pg_cursor.description]
    print(columns)
    return pd.DataFrame(pg_cursor.fetchall(), columns=columns)


def join_two_tables(table1, table2, common_column):
    return pd.merge(table2, table1, on=common_column)


def join_api_users_with_lsq_leads():
    api_enrolled_users_query = '''select auth_user.email as "email", auth_user.username as "username", auth_user.id as "user_id", concat(auth_user.first_name, ' ' , auth_user.last_name) as "full_name", courses_course.title as "course_title" from courses_courseusermapping join auth_user on auth_user.id = courses_courseusermapping.user_id join courses_course on courses_course.id = courses_courseusermapping.course_id where courses_courseusermapping.status in (8,10)'''
    first_table = execute_query_on_db('postgres_read_replica', api_enrolled_users_query)
    lsq_prospects_query = '''select emailaddress as "email", mx_graduation_year as "graduation_year_from_lsq", mx_work_experience as "work_experience_from_lsq", mx_product_graduation_year as "graduation_year_from_product", prospectid as "prospect_id" from leadsquareleadsdata'''
    second_table = execute_query_on_db('postgres_lsq_leads', lsq_prospects_query)
    joined_data = join_two_tables(first_table, second_table, 'email')
    return joined_data


def join_lsq_prospects_with_sales_call_analysis_ai():
    ds_info_query = '''select user_extracted_information -> 'current_ctc' as "current_ctc_ai",
user_extracted_information -> 'expected_ctc' as "expected_ctc_ai",
user_extracted_information -> 'graduation_year' as "graduation_year_ai",
user_extracted_information -> 'current_employer' as "current_employer_ai",
user_extracted_information -> 'is_working_professional' as "is_working_professional_ai",
user_extracted_information -> 'years_of_work_experience' as "years_of_work_experience_ai",
meta_data -> 'activity_id' as "activity_id",
meta_data -> 'prospect_activity_id' as "prospect_activity_id",
meta_data -> 'related_prospect_id' as "prospect_id"
from openai_transcribeaudio'''
    first_table = join_api_users_with_lsq_leads()
    second_table = execute_query_on_db('postgres_newton_ds', ds_info_query)
    joined_data = join_two_tables(first_table, second_table, "prospect_id")
    return joined_data


def dump_joined_data_in_results_db(**kwargs):
    def clean_input(data_type, data_value):
        if data_type == 'string':
            return 'null' if not data_value else f'\"{data_value}\"'
        elif data_type == 'datetime':
            return 'null' if not data_value else f'CAST(\'{data_value}\' As TIMESTAMP)'
        else:
            return data_value

    pg_hook = PostgresHook(postgres_conn_id='postgres_result_db')
    pg_conn = pg_hook.get_conn()
    pg_cursor = pg_conn.cursor()
    ti = kwargs['ti']
    transform_data_output = ti.xcom_pull(task_ids='join_python_data')

    for i, transform_row in transform_data_output.iterrows():
        pg_cursor.execute(
                'INSERT INTO lsq_leads_joined_data (user_id, email, username, full_name, graduation_year_from_lsq, work_experience_from_lsq, graduation_year_from_product, prospect_id, current_ctc_ai, expected_ctc_ai, graduation_year_ai, current_employer_ai, is_working_professional_ai, years_of_work_experience_ai)'
                'VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'
                'on conflict (user_id) do update set '
                'email=EXCLUDED.email,'
                'username = EXCLUDED.username,'
                'graduation_year_from_lsq=EXCLUDED.graduation_year_from_lsq,'
                'prospect_id= EXCLUDED.prospect_id,'
                'current_ctc_ai=EXCLUDED.current_ctc_ai,'
                'expected_ctc_ai=EXCLUDED.expected_ctc_ai,'
                'graduation_year_ai=EXCLUDED.graduation_year_ai,'
                'graduation_year_from_product=EXCLUDED.graduation_year_from_product;',
                (
                    transform_row['user_id'],
                    transform_row['email'],
                    transform_row['username'],
                    transform_row['full_name'],
                    transform_row['graduation_year_from_lsq'],
                    transform_row['work_experience_from_lsq'],
                    transform_row['graduation_year_from_product'],
                    transform_row['prospect_id'],
                    transform_row['current_ctc_ai'],
                    transform_row['expected_ctc_ai'],
                    transform_row['graduation_year_ai'],
                    transform_row['current_employer_ai'],
                    transform_row['is_working_professional_ai'],
                    transform_row['years_of_work_experience_ai']
                 )
        )
    pg_conn.commit()


dag = DAG(
    'lsq_leads_data',
    default_args=default_args,
    description='LSQ Leads DAG',
    schedule_interval='0 17 * * *',
    catchup=False
)


create_table = PostgresOperator(
    task_id='create_table',
    postgres_conn_id='postgres_result_db',
    sql='''CREATE TABLE IF NOT EXISTS lsq_leads_joined_data (
            user_id bigint not null PRIMARY KEY,
            email varchar(256),
            username varchar(256),
            full_name varchar(256),
            graduation_year_from_lsq varchar(256),
            work_experience_from_lsq varchar(256),
            graduation_year_from_product varchar(256),
            prospect_id varchar(256),
            current_ctc_ai varchar(256),
            expected_ctc_ai varchar(256),
            graduation_year_ai varchar(256),
            current_employer_ai varchar(256),
            is_working_professional_ai varchar(256),
            years_of_work_experience_ai varchar(256)
        );
    ''',
    dag=dag
)

join_python_data = PythonOperator(
    task_id='join_python_data',
    python_callable=join_lsq_prospects_with_sales_call_analysis_ai,
    provide_context=True,
    dag=dag
)

extract_python_data = PythonOperator(
    task_id='extract_python_data',
    python_callable=dump_joined_data_in_results_db,
    provide_context=True,
    dag=dag
)

create_table >> join_python_data >> extract_python_data
