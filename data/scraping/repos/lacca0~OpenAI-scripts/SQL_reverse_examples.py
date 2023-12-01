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

sourceFile = open('results_reverse', 'a')
prompt = []
model = ["text-davinci-003","text-curie-001","text-babbage-001","text-ada-001", "code-davinci-002", "code-cushman-001"]

prompt.append("""
SELECT City, COUNT(CustomerID) AS CustomersCount
FROM Customers
GROUP BY City
ORDER BY CustomersCount DESC
LIMIT 1;
-- The query above
""")

prompt.append("""
CREATE TABLE Person (ID INTEGER PRIMARY KEY, Name VARCHAR(100), Age INT) AS NODE;
CREATE TABLE friends (StartDate date) AS EDGE;
-- Find friends of John:

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
                stop=["\n\n"]
            )

            print("Model: " + model[j] + "\n")
            print("Temperature: %.1f \n" % k)

            if 'choices' in response:
                print(color.DARKCYAN + prompt[i] + color.END + response['choices'][0]['text'] + "\n\n")
                print(color.DARKCYAN + prompt[i] + color.END + response['choices'][0]['text'] + "\n\n", file = sourceFile)





