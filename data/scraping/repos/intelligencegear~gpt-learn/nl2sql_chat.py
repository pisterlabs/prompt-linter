import os
import openai

from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

import openai

prompt = """
Postgres SQL tables, with their properties:
Employee(id, name, department_id)
Department(id, name, address)
Salary_Payments(id, employee_id, amount, date)

Please give a query to list the names of the departments which employed more than 10 employees in the last 3 months
"""

response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "As a senior analyst, given the above schemas and data, write a detailed and correct Postgres sql query to answer the analytical question."},
        {"role": "user", "content": prompt},
    ]
)

print(response["choices"][0]["message"]["content"])
# {
#   "content": "\n\n```\nSELECT d.name as department_name\nFROM Department d\nINNER JOIN Employee e ON d.id = e.department_id\nINNER JOIN (\n  SELECT employee_id\n  FROM Salary_Payments\n  WHERE date >= NOW() - INTERVAL '3 months'\n  GROUP BY employee_id\n  HAVING COUNT(*) > 10) sub\nON e.id = sub.employee_id;\n```\n\nExplanation:\n- We begin by selecting the `name` field from the `Department` table.\n- We then join the `Department` table with the `Employee` table on the `id` field, to obtain information on the employees within each department.\n- Next, we join the resulting table with a subquery that selects the `employee_id` field from the `Salary_Payments` table for payments made in the last 3 months and groups them by `employee_id`. The subquery also filters the results to only include those with more than 10 salary payments.\n- Finally, we filter the results of the join by matching employee `id` from the resulting table with those from the subquery (using the `ON` clause).\n\nThis query will return a list of department names that have employed more than 10 employees in the last 3 months.",
#   "role": "assistant"
# }