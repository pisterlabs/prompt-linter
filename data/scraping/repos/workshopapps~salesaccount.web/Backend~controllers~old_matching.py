#!/usr/bin/python3
""" DOCUMENT MATCHING MODULE """
import json
import pandas as pd
import warnings

from .convert_file import convert_file
from .openai_request import openai_call
from .get_sections import get_sections

warnings.filterwarnings("ignore")


def compare(df1, df2):
    response = []
    x = 0
    while x < df1.shape[0]:
        print("first doc")
        p = get_sections(df1.loc[x:])
        x = p.index.stop
        y = 0
        while y < df2.shape[0]:
            print("second doc")
            q = get_sections(df2.loc[y:])
            y = q.index.stop
            k = (p, q)
            response.append(k)
    return response


def unmatched(matched_json: str, records_table):
    """Finds the unmatched transactions in the dataframe

    Args:
        matched_json: matched json
        records_df: sales record df

    Return:
        object: json
    """
    response = [ sub['Matching_details'][0] for sub in matched_json if sub['Matching'] == 'Yes' ]
    res_df = pd.DataFrame(response)
    response = records_table[~records_table.isin(res_df)].dropna()
    response = df_to_json(response)
    return response

def gptmatch(file1, file2):
    """Matches similar transactions in two documents

    Args:
        file1: first document uploaded
        file2: second document uploaded

    Return:
        object: json
    """
    keyword = """
        Match all the details in these files content below. No title. \
        Response must just be a JSON in a list, for instance [{\Response\}]. Fill empty lists when no matches with dictionary of empty string values. \
        All key value pairs should have be double quotation. Change the date field in the JSON to proper dates
        """
    statement_table = convert_file(file1)
    statement_table = pd.read_json(statement_table)
    records_table = convert_file(file2)
    records_table = pd.read_json(records_table)
    statement_csv = statement_table.to_csv()#[:800]
    records_csv = records_table.to_csv()#[:800]
    total_tokens = statement_csv + records_csv
    if len(total_tokens) > 2000:
        sections = compare(statement_table, records_table)
        print(len(sections), len(statement_csv), len(records_csv))
        response_sections = []
        for i in sections:
            result = (len(i[0].to_csv()), len(i[1].to_csv()))#arrange(i[0], i[1], keyword)
            response_sections.append(result)
        print(response_sections)
        # print('===================')
        # matched_response = [item for sublist in response_sections for item in sublist]
        # # unmatched_response = unmatched(matched_response, records_table)
        # # print("******************")
        # # print(matched_response)
        # return matched_response#, unmatched_response]
    else:
        columns_a = list(statement_table.columns)
        columns_b = list(records_table.columns)
        example = "Example\n[\n{"
        for x in columns_a:
            example += f"\n    \"{x}\":"

        example += "\n   \"Matching\": \"Yes\" or \"No\"\n   \"Matching_details\":\n   [\n   {"
        for x in columns_b:
            example += f"\n    \"{x}\": "
        example += "\n   }\n   ]\n}\n]"
        # print(example)
        prompt = f"{keyword}\n\n{example}\n\n{statement_csv}\n\n{records_csv}\n\n"

        response = openai_call(prompt, 0.05)
        # try:
        matched_response = json.loads(response) 
        # unmatched_response = unmatched(matched_response, records_table)
        return matched_response#, unmatched_response]
        # except:
        #     index = response.index('[')
        #     matched_response = json.dumps(response[index:])
        #     # unmatched_response = unmatched(matched_response, records_table)
        # return matched_response

def arrange(statement_table, records_table, keyword):
    statement_csv = statement_table.to_csv()
    records_csv = records_table.to_csv()
    columns_a = list(statement_table.columns) 
    columns_b = list(records_table.columns)
    example = "Example\n[\n{"
    for x in columns_a:
        example += f"\n    \"{x}\":"

    example += "\n   \"Matching\": \"Yes\" or \"No\"\n   \"Matching_details\":\n   [\n   {"
    for x in columns_b:
        example += f"\n    \"{x}\": "
    example += "\n   }\n   ]\n}\n]"

    prompt = f"{keyword}\n\n{example}\n\n{statement_csv}\n\n{records_csv}\n\n"

    response = openai_call(prompt, 0.05)
    try:
        print(response)
        matched_response = response
        return matched_response
    except:
        index = response.index('\n')
        matched_response = eval(response[index+1:])
        return matched_response
