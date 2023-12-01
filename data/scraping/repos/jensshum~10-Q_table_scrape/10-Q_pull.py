# -*- coding: utf-8 -*-
"""

SEC Filing Scraper
@author: AdamGetbags

"""

# import modules
import requests
import pandas as pd
import sys
from bs4 import BeautifulSoup
import os
import openai
import time

from format_AB_private_investor import format_AB_private_investor
from format_arcc import format_arcc

LLM = 'gpt-4'
# create request header
headers = {'User-Agent': "email@address.com"}

# FUNCTIONS ---------------------------------------------------------

def get_accession_nums_and_html(cik):
    """
    Gets accession number and html cap to be used in accessing the corresponding 10-Q url.
    """
    filingMetadata = requests.get(
        f'https://data.sec.gov/submissions/CIK{cik}.json',
        headers=headers
        )
    # dictionary to dataframe
    allForms = pd.DataFrame.from_dict(
                filingMetadata.json()['filings']['recent'])

    # review columns
    allForms.columns
    allForms[['accessionNumber', 'reportDate', 'form']].head(100)

    # 10-Q metadata
    allForms.iloc[11]
    forms = allForms[allForms['form'] == '10-Q'].reset_index(drop = True)

    accession_numbers = forms.loc[forms["form"] == "10-Q", "accessionNumber"].tolist()
    html_caps = forms.loc[forms["form"] == "10-Q", "primaryDocument"].tolist()
    return accession_numbers, html_caps

def get_url(cik,num,html):
    """
    Gets the url to be used for extracting the html from the 10-Q.
    """
    num = num.replace("-","")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{num}/{html}"
    return url

def get_html(url):
    """
    Gets the html content to be extracted from the 10-Q url.
    """
    response = requests.get(url, headers=headers)
    html_content = response.text
    return html_content



# MAIN --------------------------------------------------------------


# get all companies data
companyTickers = requests.get(
    "https://www.sec.gov/files/company_tickers.json",
    headers=headers
    )

# review response / keys
# print(companyTickers.json().keys())

# format response to dictionary and get first key/value
firstEntry = companyTickers.json()['0']

# parse CIK // without leading zeros
directCik = companyTickers.json()['0']['cik_str']

# dictionary to dataframe
companyData = pd.DataFrame.from_dict(companyTickers.json(),
                                     orient='index')

# add leading zeros to CIK
companyData['cik_str'] = companyData['cik_str'].astype(
                           str).str.zfill(10)

# review data
company_data_dict = companyData.set_index('cik_str')['ticker'].to_dict()


cik_tickers = ["1634452", "1578620","1278752","1287750","1633858","1655050",	"1379785","1326003","1370755","1736035","1490927","17313","1571329","1534254","1578348","1617896","1633336","878932", "1513363", "1495584","1701724","1501729",
"1637417","1525759","1579412","1422183","1509892","1143513","1321741", "1572694","1683074","1674760","1476765","1715268","1627515","1675033","1509470","1618697","1661306","1559909", "1280784","1535778","1487428","1550913","1396440",
"1490349","1512931","1742313","1099941","1496099","1588272","1414932","1577791", "1744179", "1487918","1297704", "1655888", "1655887", "1747777", "1259429", "1626899", "1383414","1504619", "1372807", "845385", "1287032","81955", "1653384", 
"1743415", "1523526", "1614173", "1418076",	"1508171", "1551901", "1702510", "1544206", "1603480", "1715933", "1577134", "1464963","1521945","1508655","1580345", "1717310", "1557424","1642862", "1552198"]

cik_tickers = [value.zfill(10) for value in cik_tickers]

# # get company specific filing metadata

html_links = []
for cik in cik_tickers:
    
    accession_numbers, html_caps = get_accession_nums_and_html(cik)
    # For testing:
    if "1287750" not in cik:
        continue
    k = 1
    for html, num in zip(html_caps, accession_numbers):
        
        url = get_url(cik,num,html)
        print(url)

        html_content = get_html(url)

        # Checks html_content for the correct tables.
        investment_tables = html_content.lower().split("schedule of investments")
        if len(investment_tables) < 2:
            print("No schedule info")
            continue
        
        tables = []
        df = pd.DataFrame()
        for i, table_text in enumerate(investment_tables[1:]):
            values = []
            soup = BeautifulSoup(table_text, 'html.parser')
            table = soup.find('table')
            if table == None:
                continue
            for row in table.find_all('tr'):
                columns = row.find_all('td')
                row_data = {}
                for j, col in enumerate(columns):
                    cell_text = col.get_text(strip=True)
                    row_data[f'col{j}'] = cell_text
                values.append(row_data)
                
            df_to_append = pd.DataFrame(values)
            df_to_append = df_to_append[2:]
            if "1287750" in cik:
                # print(len(df_to_append.columns))
                if len(df_to_append.columns) < 20:
                    continue
            elif "1634452" in cik:
                if len(df_to_append.columns) > 24:
                    continue

            df = pd.concat([df, df_to_append], ignore_index=True)

        print(cik)
        
        if "1634452" in cik:
            break
            df = format_AB_private_investor(df)
        elif "1287750" in cik:
            
            df = format_arcc(df)
            
            try:
                os.mkdir(f"CIK_{cik}")
            except Exception:
                pass
            df.to_csv(f"CIK_{cik}/CIK_{cik}_filenum_{k}_of_{len(accession_numbers)}_test.csv", index=False)
            k += 1
            sys.exit()
        break

    # Specific to Blue Owl Capital
    # df['Reference Rate'] = df['Interest'].apply(lambda x: x.split(' ')[0])
    # df['Spread'] = df['Interest'].apply(lambda x: x.split(' ')[1])
    # df['Spread'] = df['Interest'].apply(lambda x: x.split(' ')[1])
    # del df['Interest']


# https://www.sec.gov/Archives/edgar/data/1287750/000128775023000036/arcc-20230630.htm