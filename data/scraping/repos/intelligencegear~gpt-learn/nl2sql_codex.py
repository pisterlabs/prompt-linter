import os
import openai

from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

prompt = """
### Postgres SQL tables, with their properties:
#
# Employee(id, name, department_id)
# Department(id, name, address)
# Salary_Payments(id, employee_id, amount, date)
#
### A query to list the names of the departments which employed more than 10 employees in the last 3 months
SELECT
"""

if __name__ == '__main__':
  response = openai.Completion.create(
    model="code-davinci-002",
    prompt=prompt,
    temperature=0,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stop=["#", ";"]
  )

  print(response["choices"][0]["text"])
#     d.name
# FROM
#     Department d
#     INNER JOIN Employee e ON d.id = e.department_id
#     INNER JOIN Salary_Payments sp ON e.id = sp.employee_id
# WHERE
#     sp.date >= DATE_SUB(CURDATE(), INTERVAL 3 MONTH)
# GROUP BY
#     d.name
# HAVING
#     COUNT(e.id) > 10