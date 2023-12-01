import os
import pandas as pd
import openai
from json import loads, dumps
from dotenv import load_dotenv


load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

df = pd.read_csv('predicted_anomalies.csv')
result = df.to_json(orient="split")
parsed = loads(result)

completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
      {"role": "system", "content": "You're a pro at finding anomalies in a basketball game."},
      {"role": "user", "content": f"This data is anomalous, that is, it stands out from other data. Tell me, why "
                                  f"can they be anomaly? {dumps(parsed, indent=4)}"}
  ]
)


my_file = open('why_anomalies_gpt.txt', 'a+')
my_file.write(f'\n\n{completion.choices[0].message.content}')
my_file.close()
print(completion.choices[0].message.content)
