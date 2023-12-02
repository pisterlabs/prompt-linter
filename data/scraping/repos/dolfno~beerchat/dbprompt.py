from langchain.prompts.prompt import PromptTemplate

_DEFAULT_TEMPLATE = """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"

Only use table {table_info}.
daily_production_count is in units of beer.

USE MAX, INSTEAD OF SUM BECAUSE VALUES ARE ALREADY SUMMED.
Question: {input}"""
prompt_template = PromptTemplate(input_variables=["input", "table_info", "dialect"], template=_DEFAULT_TEMPLATE)

TABLE_INFO = """
CREATE TABLE reporting.datapoints (
	id SERIAL NOT NULL, 
	timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	name VARCHAR NOT NULL, 
	value BIGINT NOT NULL, 
	opco VARCHAR, 
	site VARCHAR, 
	work_center VARCHAR, 
	last_updated TIMESTAMP WITHOUT TIME ZONE DEFAULT now() NOT NULL, 
	CONSTRAINT datapoints_pkey PRIMARY KEY (id)
)

daily_production_count is in units of beer.
opi_analytics_equipment_count is the number of machines connected to OPI.

/*
6 rows from datapoints table:
| id     | timestamp           | name                             | value  | opco    | site | last_updated               |
|--------|---------------------|----------------------------------|--------|---------|------|----------------------------|
|    334 | 2022-11-30 00:00:00 | daily_production_count           | 587179 | pl001oc | pl02 | 2022-12-02 10:00:23.986712 |
|    132 | 2022-11-30 00:00:00 | equipment                        |     11 | br001oc | br01 | 2022-12-02 10:00:23.986712 |
|   1688 | 2020-01-01 00:00:00 | measurement_count                |      0 | nl049oc | nl01 | 2022-12-02 10:00:23.986712 |
| 117411 | 2023-01-06 00:00:00 | opi_analytics_equipment_count    |     20 | pl001oc | pl03 | 2023-01-06 00:02:11.270405 |
| 117370 | 2023-01-06 00:00:00 | opi_analytics_work_centers_count |      1 | mx002oc | mx04 | 2023-01-06 00:02:11.270405 |
|    121 | 2022-11-30 00:00:00 | tags                             |    217 | br001oc | br07 | 2022-12-02 10:00:23.986712 |
*/
"""
