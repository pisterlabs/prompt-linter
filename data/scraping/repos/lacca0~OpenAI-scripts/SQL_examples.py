import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

sourceFile = open('results', 'a')
prompt = []
model = ["text-davinci-003","text-curie-001","text-babbage-001","text-ada-001", "code-davinci-002", "code-cushman-001"]

prompt.append("""
-- =============================================
-- ...
-- Parameters:
--   Station properties:
--     @name (string)
--     @lat (double)
--     @long (double)
--     @dock_count (int)
--     @city (string)
--     @installation_date (string)
--     @id (int)
--   Trip properties:
--     @id
--     @duration (long)
--     @start_date (string)
--     @start_station_name (string)
--     @start_station_id (long)
--     @end_date (string)
--     @end_station_name (string)
--     @end_station_id (long) 
--     @bike_id (long)
--     @subscription_type (string)
--     @zip_code (string)
-- Returns: a merged table with the city indicators of the start and end city
-- =============================================
SELECT
""")

prompt.append("""
--  We have a graph with the following node properties: name, age, occupation, address. 
--  It has the following edge properties: relationship_type, added_date, comment. 
--  We want to find the chef with the most friends. 
--  Please create a SQL query that takes the "nodes" and "edges" table and returns the result.
SELECT
""")

prompt.append("""
-- We have a table with the following properties: CustomerID, CustomerName, ContactName, Address, City, PostalCode, Country.
-- We need to find a city with the most customers.
SELECT
""")


for  i in range (len(prompt)):
    for j in range (len(model)):
        k_iter = [k * 0.5 for k in range(0, 5)]
        for k in k_iter:
            response = openai.Completion.create(
                model=model[j],
                prompt=prompt[i],
                temperature=k,
                max_tokens=1000,
                top_p=1,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=["--"]
            )

            print("Model: " + model[j] + "\n")
            print("Temperature: %.1f \n" % k)

            if 'choices' in response:
                print(color.DARKCYAN + prompt[i] + color.END + response['choices'][0]['text'] + "\n\n")
                print(color.DARKCYAN + prompt[i] + color.END + response['choices'][0]['text'] + "\n\n", file = sourceFile)
