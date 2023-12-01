import pandas as pd
from bs4 import BeautifulSoup
import requests
from openai import OpenAI
from tqdm import tqdm
import time
from datetime import datetime

def get_completion(prompt, model="gpt-4", client=OpenAI()):
    response =  client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=model
                )
    return response.choices[0].message.content

def classify_dates(text, client):
    prompt=f"""
        Classify the text (descriptions of academic calendar dates) delimited by triple backticks into one of the following categories: ["Class", "Exam", "Holiday"].\
        ```{text}```
        """
    cat = get_completion(prompt, client)

    return cat

def helper(s):
    words = s.split()
    if len(words) < 3:
        return s
    return f"{words[0]} {words[1]} {words[-1]}"

def convert_date(date_str):
    date_obj = datetime.strptime(date_str, "%B %d %Y")
    formatted_date = date_obj.strftime("%m/%d/%Y")
    return formatted_date

def prep_data(url):
    print(url)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    table = soup.find('table')

    table_data = []
    for row in table.find_all('tr'):
        row_data = []
        for cell in row.find_all(['th', 'td']):
            row_data.append(cell.get_text())
        table_data.append(row_data)
        
    df = pd.DataFrame(table_data)

    if (df.shape[1]) == 2:
        df.columns = ["daydate", "desc"]

        df = df[df['daydate'].str.split().str.len() <= 3]

        df['len'] = df['daydate'].str.split().str.len()

        
        new_df = pd.DataFrame()

        for index, row in df.iterrows():
            if len(row['daydate'].split()) == 2:
                day = row['daydate']

                new_df = pd.concat(
                        [new_df, pd.DataFrame({'day': [day], 'date': ['\xa0'], 'desc': ['\xa0']})]
                        )
            try:

                day = row['daydate'].split(',')[0]
            
                date = row['daydate'].split(',')[1].strip()

                desc = row['desc']

                new_df = pd.concat(
                        [new_df, pd.DataFrame({'day': [day], 'date': [date], 'desc': [desc]})]
                        )

            except:
                pass
        
        df = new_df
    else:                                               
        df.columns = ["date", "day", "desc"]

    df = df[['day', 'date', 'desc']]
    
    current_year = None

# Iterate through the rows of the DataFrame
    for index, row in df.iterrows():
        if row['desc'] == '\xa0':
            # Extract the year from the "Date" column and store it as the current year
            current_year = int(row['day'].split()[1])

        else:
            # If the row is not a "Mini Header," update the "Date" column with the current year
            df['date'] = df['date'] + f' {current_year}'

    df['date'] = df['date'].apply(helper)

    # Filter out the rows with '\xa0' in both "Day" and "Desc" columns
    df = df[(df['desc'] != '\xa0')]

    df['date'] = df['date'].str.replace('None', '')

    mask = df['desc'].str.contains("reading period|Forum|Founder's Weekend|Founders' Weekend")

    df = df[~mask]

    df.reset_index(drop=True, inplace=True)

    df['date_string'] = df['date'].apply(convert_date)

    return df

def run_classification(client):
    years = ["2017-2018", "2018-2019", "2019-2020", "2020-2021", "2021-2022"]

    for year in years[:1]:
        
        url = f"https://registrar.duke.edu/{year}-academic-calendar/"

        year_df = prep_data(url)

        cats = []

        for idx, row in tqdm(year_df.iterrows()):
            desc = row['desc']

            cat = classify_dates(desc, client)

            time.sleep(5)

            cats.append(cat)

        year_df['cat'] = cats

        year_df.to_csv(f"{year}_dates.csv", index=False)

        print(f"Year {year} Done\n")

if __name__ == "__main__":
    client = OpenAI(
    api_key="API_KEY",
)
    run_classification(client)
