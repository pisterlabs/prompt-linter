from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from agents.zero_shot import ZeroShotAgent

FEW_SHOT_PROMPT = """
"Database schema in the form of CREATE_TABLE statements:
{database_schema}

Here are a few examples, \"Q\" represents the question and \"A\" represents the corresponding SQL-query :
Q: List out the account numbers of female clients who are oldest and has lowest average salary, calculate the gap between this lowest average salary with the highest average salary?

A: SELECT T1.account_id , ( SELECT MAX(A11) - MIN(A11) FROM district ) FROM account AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id WHERE T2.district_id = ( SELECT district_id FROM client WHERE gender = 'F' ORDER BY birth_date ASC LIMIT 1 ) ORDER BY T2.A11 DESC LIMIT 1
Q: For the branch which located in the south Bohemia with biggest number of inhabitants, what is the percentage of the male clients?

A: SELECT CAST(SUM(T1.gender = 'M') AS REAL) * 100 / COUNT(T1.client_id) FROM client AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id WHERE T2.A3 = 'south Bohemia' GROUP BY T2.A4 ORDER BY T2.A4 DESC LIMIT 1
Q: \"For the client who first applied the loan in 1993/7/5, what is the increase rate of his/her account balance from 1993/3/22 to 1998/12/27?
A: SELECT CAST((SUM(IIF(T3.date = '1998-12-27', T3.balance, 0)) - SUM(IIF(T3.date = '1993-03-22', T3.balance, 0))) AS REAL) * 100 / SUM(IIF(T3.date = '1993-03-22', T3.balance, 0)) FROM loan AS T1 INNER JOIN account AS T2 ON T1.account_id = T2.account_id INNER JOIN trans AS T3 ON T3.account_id = T2.account_id WHERE T1.date = '1993-07-05'

Using valid SQL, answer the following question based on the tables provided above.
It is important to use qualified column names in the SQL-query, meaning the form \"SELECT table_name.column_name FROM table_name;

Hint helps you to write the correct sqlite SQL query.
Question: {question}
Hint: {evidence}
DO NOT return anything else except the SQL query."
"""



class FewShotAgent(ZeroShotAgent):

    def __init__(self, llm):        
        self.llm = llm

        self.prompt_template = FEW_SHOT_PROMPT
        prompt = PromptTemplate(
            input_variables=["question", "database_schema", "evidence"],
            template=self.prompt_template,
        )

        self.chain = LLMChain(llm=llm, prompt=prompt)

