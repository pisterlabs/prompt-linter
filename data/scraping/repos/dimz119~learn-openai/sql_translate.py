import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  model="code-davinci-002",
  prompt="### Postgres SQL tables, with their properties:\n#\n# Employee(id, name, department_id)\n# Department(id, name, address)\n# Salary_Payments(id, employee_id, amount, date)\n#\n### A query to list the names of the departments which employed more than 10 employees in the last 3 months\nSELECT",
  temperature=0,
  max_tokens=150,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0,
  stop=["#", ";"] # When to stop
)
print(response)

# SELECT DISTINCT department.name
# FROM department
# INNER JOIN employee ON department.id = employee.department_id
# INNER JOIN salary_payments ON employee.id = salary_payments.employee_id
# WHERE salary_payments.date >= (CURRENT_DATE - INTERVAL '3 months')
# GROUP BY department.name
# HAVING COUNT(employee.id) > 10
