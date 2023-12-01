import openai
from datetime import date, timedelta
import json
import psycopg2

from dict_values import adiq_db_creds
from utils import *

openai.api_key = open('openai_key.txt', 'r').read().strip()

def get_db_connection():

    creds = adiq_db_creds

    conn = psycopg2.connect(
        host=creds['host'],
        port=creds['port'],
        dbname=creds['db'],
        user=creds['username'],
        password=creds['password']
    )
    return conn

def get_daterange():

    yesterday = str(date.today() - timedelta(days=1))
    yesterday = "'"+yesterday+"'"

    start_date = "'2023-07-01'"
    end_date = yesterday

    return start_date, end_date

start_date, end_date = get_daterange()
conn = get_db_connection()

Databases = listDatabases()
Tables = listTables()
TableDetails = getTableDetails()
TableRowSamples = getTableRowSamples()
DatabaseExampleResponses = getDatabaseExampleResponses()
DatabaseSpecialInstructions = getDatabaseSpecialInstructions()

message_history = [{"role": "user", 
                    "content": f"""You are a SQL query generator bot for postgreSQL database. 
                    That means you will generate SQL queries for a postgreSQL database.

                    Following json contains the list of databases and their brief descriptions:
                    {Databases}

                    The database contains the following tables:
                    {Tables}

                    The following json contains the details of the table:
                    {TableDetails}
                    
                    Some sample rows of the table:
                    {TableRowSamples}

                    Some sample natural language questions and their corresponding SQL queries:
                    {DatabaseExampleResponses}
                    
                    Some special instructions:
                    {DatabaseSpecialInstructions}

                    ---------------------------------------------

                    I will ask a question about the data in natural language in my message, and you will reply only with a SQL query 
                    that will generate the necessary response when performed on the postgreSQL database. 
                    Reply only with the sql query to further input. If you understand, say OK."""},
                   {"role": "assistant", 
                    "content": f"OK"}]

def generate_query(question):
    global message_history

    message_history.append({"role": "user", "content": question})

    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=message_history,
    )

    reply_content = completion.choices[0].message.content
    message_history.append({"role": "assistant", "content": reply_content})

    return reply_content

def execute_query(ext_query):

    # Create a cursor object to interact with the database
    cursor = conn.cursor()

    base_query = f"""

    SET TIMEZONE TO 'Asia/Dhaka';

    WITH "on_beats" as (
        SELECT
            *
        FROM
            "campaign_events_aggregated_heartbeat_view"
        WHERE
            "day_start" >= {start_date}::date
            AND "day_start" < {end_date}::date + INTERVAL '1 day'
    ), "play_time" as (
        SELECT
            "bundle_number",
            "day_start",
            EXTRACT(epoch FROM uptime("on_beat")) as "total_play_time",
            EXTRACT(epoch FROM downtime("on_beat")) as "total_non_play_time"
        FROM "on_beats"
    ), "online_beats" as (
        SELECT
            *
        FROM
            "bundle_online_heartbeat_aggregated_view"
        WHERE
            "day_start" >= {start_date} :: date
            AND "day_start" < {end_date} :: date + INTERVAL '1 day'
    ), "online_time" as (
        SELECT
            "bundle_number",
            "day_start",
            EXTRACT(epoch FROM uptime("online_beat")) as "total_online_time",
            EXTRACT(epoch FROM downtime("online_beat")) as "total_offline_time"
        FROM "online_beats"
    ), "raw_data" as (
        SELECT
            "on_beats"."bundle_number",
            "on_beats"."day_start" as "day_start",
            "on_beats"."opening" as "start_time",
            "on_beats"."closing" as "end_time",
            "play_time"."total_play_time" as "total_play_time",
            GREATEST(
                "play_time"."total_play_time" - "online_time"."total_online_time",
                0
            ) as "total_offline_time",
            GREATEST(
                "play_time"."total_non_play_time" - EXTRACT(epoch FROM (INTERVAL '1 day' - ("on_beats"."closing" - "on_beats"."opening")))::integer,
                0
            ) as "off_time",
            EXTRACT(epoch FROM INTERVAL '10 hours')::integer as "expected_play_time",
            "play_time"."total_play_time" / EXTRACT(epoch FROM INTERVAL '10 hours')::integer as "efficiency"
        FROM "on_beats"
        LEFT OUTER JOIN "play_time"
            ON "on_beats"."bundle_number" = "play_time"."bundle_number"
            AND "on_beats"."day_start" = "play_time"."day_start"
        LEFT OUTER JOIN "online_time"
            ON "on_beats"."bundle_number" = "online_time"."bundle_number"
            AND "on_beats"."day_start" = "online_time"."day_start"
    ), "dates" as (
        SELECT
            {start_date}::date + (i || ' days')::interval as "day_start"
        FROM generate_series(0, {end_date}::date - {start_date}::date) i
    ), "bundles" as (
        SELECT
            id as "bundle_id",
            document->>'bundle_number' as "bundle_number"
        FROM raw_bundles
    ), "dailyLog" as (
        SELECT
            raw_zones.document->>'name' as "zone_name",
            raw_areas.document->>'name' as "area_name",
            raw_points.document->>'code' as "point_code",
            raw_points.document->>'name' as "point_name",
            bundles.bundle_number as "bundle_number",
            "dates"."day_start"::date as "Date",
            "raw_data"."start_time" as "Started",
            end_time as "Ended",
            total_play_time as "total_play_time_seconds",
            ROUND(total_play_time/60) as "total_playtime_minutes",
            TO_CHAR((total_play_time || ' seconds')::interval, 'HH24:MI:SS') as "Total Play Duration",
            efficiency as "Efficiency",
            total_offline_time as "total_offline_time_seconds",
            TO_CHAR((total_offline_time || ' seconds')::interval, 'HH24:MI:SS') as "Offline Play Duration",
            off_time as "off_time_seconds",
            TO_CHAR((off_time || ' seconds')::interval, 'HH24:MI:SS') as "Box Off Duration",
            CASE
                WHEN start_time IS NOT NULL THEN 'OPENED'
                ELSE 'NOT-OPENED'
            END as "Opened"
        FROM bundles
        RIGHT OUTER JOIN dates ON TRUE
        LEFT OUTER JOIN raw_data
            ON bundles.bundle_number = raw_data.bundle_number
            AND "raw_data"."day_start" = "dates"."day_start"
        LEFT OUTER JOIN "raw_points"
            ON "raw_points".document->>'bundle_id' = "bundles"."bundle_id"
        LEFT OUTER JOIN "raw_areas"
            ON "raw_areas".id = "raw_points".document->>'area_id'
        LEFT OUTER JOIN "raw_zones"
            ON "raw_zones".id = "raw_areas".document->>'zone_id'
        ORDER BY
            dates.day_start::date,
            bundles.bundle_number
    )

    """

    full_query = f"""

        {base_query}

        {ext_query}
    """

    #######################################
    cursor.execute(full_query)

    # Fetch all the rows returned by the query
    results = cursor.fetchall()

    # Fetch column names
    column_names = [desc[0] for desc in cursor.description]

    # Render the output
    print("| " + " | ".join(column_names) + " |")
    print("|-" + "-|-".join(["-" * len(name) for name in column_names]) + "-|")
    for row in results:
        print("| " + " | ".join(map(str, row)) + " |")

while True:
    print()
    question = input("Human: ")

    print()
    print("Bot: \n")
    query = generate_query(question)
    print("Query:\n"+ query)
    print("\nAnswer:")

    # execute_query(query)

    try:
        execute_query(query)
    except Exception as e:
        # print(e)
        print("Invalid question. Please try again.")