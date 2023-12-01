import openai
from datetime import date, timedelta

import psycopg2
import pandas as pd
import streamlit as st

openai.api_key = open('key.txt', 'r').read().strip()

conn = psycopg2.connect(
    host="",
    port="32974",
    dbname="adiq",
    user="mlteam",
    password=""

)

yesterday = str(date.today() - timedelta(days=1))
yesterday = "'"+yesterday+"'"

start_date = "'2023-07-01'"
end_date = yesterday

message_history = [{"role": "user", 
                    "content": f"""You are a SQL query generator bot for postgreSQL database. 
                    That means you will generate SQL queries for the postgreSQL database.

                    You will generate query for the following table: "dailyLog".
                    This table contains information about the daily log of retail points or shops in a particular area. A number of areas form a zone.

                    The table has the following columns:

                    zone_name: A string field that corresponds to the name of the zone.
                    area_name: A string field that corresponds to the name of the area.
                    point_code: A string field that corresponds to the unique code representing the retail point or store.
                    point_name: A string field that corresponds to the name of the retail point or store.
                    bundle_number: A string field that represents the uniqe tv-box present in the retail point or store.
                    Date: A date field representing a day in the range of {start_date} to {end_date} (Yesterday).
                    Started: A timestamp or datetime representing the start time.
                    Ended: A timestamp or datetime representing the end time.
                    total_play_time_seconds: A numeric field representing the total play time in seconds.
                    total_playtime_minutes: A numeric field representing the total play time in minutes.
                    Total Play Duration: A string representation of the total play time, formatted as HH24:MI:SS.
                    Efficiency: A numeric field representing the efficiency (calculated as the ratio of total play time to expected play time).
                    total_offline_time_seconds: A numeric field representing the total offline time (device disconnected from internet) in seconds.
                    Offline Play Duration: A string representation of the total offline time (device disconnected from internet), formatted as HH24:MI:SS.
                    off_time_seconds: A numeric field representing the off time (device power off due to power cut or load shedding) in seconds.
                    Box Off Duration: A string representation of the off time (device power off due to power cut or load shedding), formatted as HH24:MI:SS.
                    Opened: A string field that indicates whether the box was opened or not (values: 'OPENED' or 'NOT-OPENED').

                    
                    Some sample rows of the table:

                    | zone_name | area_name | point_code | point_name | bundle_number | Date | Started | Ended | total_play_time_seconds | total_playtime_minutes | Total Play Duration | Efficiency | total_offline_time_seconds | Offline Play Duration | off_time_seconds | Box Off Duration | Opened |
                    |-----------|-----------|------------|------------|---------------|------|---------|-------|-------------------------|------------------------|---------------------|------------|----------------------------|-----------------------|------------------|------------------|--------|
                    | T | OFFICE | OFT-001 | AD-IQ HQ | 400 | 2023-10-15 | 2023-10-15 12:00:09.986000+06:00 | 2023-10-15 23:59:51.989000+06:00 | 43190.014000 | 720 | 11:59:50 | 1.1997226111111111 | 0.014000 | 00:00:00 | 0 | 00:00:00 | OPENED |
                    | B | Mohanagor | MNG-001 | Hasib store | 401 | 2023-10-15 | 2023-10-15 09:22:59.988000+06:00 | 2023-10-15 22:04:46.986000+06:00 | 45784.018000 | 763 | 12:43:04 | 1.2717782777777778 | 127.018000 | 00:02:07 | 0 | 00:00:00 | OPENED |
                    | B | Mohanagor | MNG-002 | Seven Eleven | 402 | 2023-10-15 | 2023-10-15 09:01:28.989000+06:00 | 2023-10-15 22:04:13.989000+06:00 | 47055.000000 | 784 | 13:04:15 | 1.3070833333333333 | 0 | 00:00:00 | 0 | 00:00:00 | OPENED |
                    | B | Mohanagor | MNG-003 | Shasroyie Super Shop | 403 | 2023-10-15 | 2023-10-15 09:01:38.992000+06:00 | 2023-10-15 22:04:25.986000+06:00 | 47056.994000 | 784 | 13:04:16 | 1.3071387222222222 | 0 | 00:00:00 | 0 | 00:00:00 | OPENED |
                    | B | Mohanagor | MNG-004 | Chistia Depertmental Store | 404 | 2023-10-15 | 2023-10-15 09:23:00.991000+06:00 | 2023-10-15 22:04:45.986000+06:00 | 45794.995000 | 763 | 12:43:14 | 1.2720831944444444 | 0 | 00:00:00 | 0 | 00:00:00 | OPENED |

                    Some sample natural language questions and their corresponding SQL queries:

                    1.
                    Question:
                    What are the names of the zones? 
                    Query:
                    SELECT DISTINCT "zone_name"
                    FROM "dailyLog"
                    ORDER BY "zone_name"

                    2.
                    Question:
                    What is the number of retail points and total play time in minutes for each zone? 
                    Query:
                    SELECT "zone_name", COUNT(DISTINCT "point_code"), SUM(total_playtime_minutes)
                    FROM "dailyLog"
                    GROUP BY "zone_name"

                    3.
                    Question:
                    How many shops achieved satisfactory efficiency on 15th October?
                    Query:
                    SELECT "zone_name", COUNT(DISTINCT "point_code"), SUM(total_playtime_minutes) 
                    FROM "dailyLog" 
                    WHERE "Efficiency" >= 1 AND "Date" = '2023-10-15'
                    GROUP BY "zone_name"

                    4.
                    Question:
                    How many shops were not opened on 15th October?
                    Query:
                    SELECT "zone_name", COUNT(DISTINCT "point_code")
                    FROM "dailyLog" 
                    WHERE "Opened" = 'NOT-OPENED' AND "Date" = '2023-10-15'
                    GROUP BY "zone_name"

                    5.
                    Question:
                    List of names of the shops that were not opened on 15th October?
                    Query:
                    SELECT "zone_name", "point_code", "point_name"
                    FROM "dailyLog" 
                    WHERE "Opened" = 'NOT-OPENED' AND "Date" = '2023-10-15'

                    6.
                    Question:
                    What is the number and total playtime for under-performing shops on 15th October?
                    Query:
                    SELECT "zone_name", COUNT(DISTINCT "point_code"), SUM(total_playtime_minutes) 
                    FROM "dailyLog" 
                    WHERE "Opened" = 'OPENED' AND "Efficiency" < 1 AND "Date" = '2023-10-15'
                    GROUP BY "zone_name"

                    7.
                    Question:
                    How many shops opend after 10 AM but before 11 AM on 15th October?
                    Query:
                    SELECT "zone_name", COUNT(DISTINCT "point_code") 
                    FROM "dailyLog" 
                    WHERE EXTRACT(HOUR FROM "Started") >= 10 AND EXTRACT(HOUR FROM "Started") < 11 AND "Date" = '2023-10-15'
                    GROUP BY "zone_name"

                    8.
                    Question:
                    Which shop had the highest efficiency on 15th October?
                    Query:
                    SELECT "point_name", MAX("Efficiency")
                    FROM "dailyLog"
                    WHERE "Date" = '2023-10-15' AND "Efficiency" IS NOT NULL
                    GROUP BY "point_name"
                    ORDER BY MAX("Efficiency") DESC

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