import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



response = client.completions.create(model="text-davinci-003",
prompt="A two-column spreadsheet of top science fiction movies and the year of release:\n\nTitle |  Year of release",
temperature=0.5,
max_tokens=60,
top_p=1.0,
frequency_penalty=0.0,
presence_penalty=0.0)
# print out the spreadsheet
print(response['choices'][0]['text'])
# put the response into a csv file with headers: Title, Year of release
writeFile = open("spreadsheet.csv", "w", encoding="utf-8")
writeFile.write("Title,Year of release\n")
writeFile.write(response['choices'][0]['text'])
writeFile.close()
