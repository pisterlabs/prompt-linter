import os
import json
import boto3
import sys
from dotenv import load_dotenv
import pandas as pd
import pypdfium2 as pdfium
import numpy as np
import openai

#Loading Environmental Variables
load_dotenv()
AWSSecretKey = os.getenv('AWSSecretKey')
AWSAccessKeyId = os.getenv('AWSAccessKeyId')
OPENAIKEY = os.getenv('OPENAIKEY')

#Loading Prepared Medical Dictionary and Priority List
f = open('../../Report Analysis/analysis.json')
report_list = json.load(f)
f.close()
g = open('../../Report Analysis/priority.json')
priority_list = json.load(g)
g.close()

# OCR Function 1
def get_rows_columns_map(table_result, blocks_map):
    rows = {}
    for relationship in table_result['Relationships']:
        if relationship['Type'] == 'CHILD':
            for child_id in relationship['Ids']:
                cell = blocks_map[child_id]
                if cell['BlockType'] == 'CELL':
                    row_index = cell['RowIndex']
                    col_index = cell['ColumnIndex']
                    if row_index not in rows:
                        rows[row_index] = {}
                    rows[row_index][col_index] = get_text(cell, blocks_map)
    return rows

# OCR Function 2
def get_text(result, blocks_map):
    text = ''
    if 'Relationships' in result:
        for relationship in result['Relationships']:
            if relationship['Type'] == 'CHILD':
                for child_id in relationship['Ids']:
                    word = blocks_map[child_id]
                    if word['BlockType'] == 'WORD':
                        text += word['Text'] + ' '
                    if word['BlockType'] == 'SELECTION_ELEMENT':
                        if word['SelectionStatus'] =='SELECTED':
                            text +=  'X '
    return text

# OCR Function 3
def get_table_csv_results(file_name):
    with open(file_name, 'rb') as file:
        img_test = file.read()
        bytes_test = bytearray(img_test)
    client = boto3.client('textract', aws_access_key_id = AWSAccessKeyId,
            aws_secret_access_key = AWSSecretKey, 
            region_name = 'ap-south-1')
    response = client.analyze_document(Document={'Bytes': bytes_test}, FeatureTypes=['TABLES'])
    blocks=response['Blocks']
    blocks_map = {}
    table_blocks = []
    for block in blocks:
        blocks_map[block['Id']] = block
        if block['BlockType'] == "TABLE":
            table_blocks.append(block)
    if len(table_blocks) <= 0:
        return "<b> NO Table FOUND </b>"
    csv = ''
    for index, table in enumerate(table_blocks):
        csv = generate_table_csv(table, blocks_map, index +1)
    return csv

# OCR Function 4
def generate_table_csv(table_result, blocks_map, table_index):
    rows = get_rows_columns_map(table_result, blocks_map)
    columns = []
    entity = []
    obs = []
    unit = []
    interval = []
    print("Processing Table: " + str(table_index))
    for row_index, cols in rows.items():
        if row_index == 1:
            for col_index, text in cols.items():
                columns.append(text.strip())
        elif "Units" in columns:
            entity.append(cols[1].strip())
            if cols[2] == '':
                obs.append(-1)
            else:
                cols[2] = cols[2].replace(',', '')
                obs.append(float(cols[2]))
            unit.append(cols[3].strip())
            if cols[4] == '':
                interval.append(-1)
            else:
                interval.append(cols[4].strip())

            data = {
                columns[0] : entity,
                columns[1] : obs,
                columns[2] : unit,
                columns[3] : interval
            }
        elif "Unit" in columns:
            entity.append(cols[1].strip())
            if cols[2] == '':
                obs.append(-1)
            else:
                cols[2] = cols[2].replace(',', '')
                obs.append(float(cols[2]))
            unit.append(cols[3].strip())
            if cols[4] == '':
                interval.append(-1)
            else:
                interval.append(cols[4].strip())

            data = {
                columns[0] : entity,
                columns[1] : obs,
                columns[2] : unit,
                columns[3] : interval
            }
        else:
            data = {}
            break
    df = pd.DataFrame(data)
    return df

# Extracting abnormal parameters
def analysis(df):
    anomalies = {}
    columns = df.columns
    for i in range(len(df)):
        parameter = df.loc[i, columns[0]]
        observed = df.loc[i, columns[1]]
        limits = df.loc[i, columns[3]]
        vals = []
        if int(observed) == -1 and int(limits) == -1:
            pass
        else:
            if "<" in limits:
                max_limit = float(limits[1:])
                min_limit = 0.0
            elif "-" in limits:
                limits = limits.split("-")
                max_limit = float(limits[1])
                min_limit = float(limits[0])
            else:
                limits = limits.split(" ")
                max_limit = float(limits[1])
                min_limit = float(limits[0])
            adj_max = float(max_limit/0.9)
            adj_min = float(((min_limit/adj_max) - 0.1)*adj_max)
            if float(observed) < adj_min:
                vals.append("low")
                vals.append(1)
            elif float(observed) > adj_max:
                vals.append("high")
                vals.append(1)
            elif float(observed) < adj_max and float(observed) > max_limit:
                vals.append("high")
                vals.append(0)
            elif float(observed) > adj_min and float(observed) < min_limit:
                vals.append("low")
                vals.append(0)
        if vals != []:
            anomalies[parameter] = vals
    return anomalies

#String Method
def toString(s): 
    string = "" 
    for element in s:
        string += element.capitalize() 
        if (element != s[len(s) - 1]):
          string += ", "
    return string

#String Matching
def longestCommonSubstring(X, Y, m, n): 
    LongestCommonArray = [[0 for k in range(n+1)] for l in range(m+1)]  
    result = 0
    for i in range(m + 1):
        for j in range(n + 1):
            if (i == 0 or j == 0):
                LongestCommonArray[i][j] = 0
            elif (X[i-1] == Y[j-1]):
                LongestCommonArray[i][j] = LongestCommonArray[i-1][j-1] + 1
                result = max(result, LongestCommonArray[i][j])
            else:
                LongestCommonArray[i][j] = 0
    return result

#Finding String
def findString(Y):
    report_list_keys = report_list.keys()
    res = 0
    res_string = ""
    for i in report_list_keys:
        m = len(i)
        n = len(Y)
        temp = longestCommonSubstring(i, Y, m, n)
        if (res < temp):
            res = temp
            res_string = i
    return res_string

# Getting Line One
def getStart(output):
    start = "From the report, it is observed that "
    i = 0
    for k,v in output.items():
        start += str(k)
        start += " is "
        start += str(v[0])
        if i == len(output) - 2:
            start += " and "
        elif i == len(output) - 1:
            start += "."
        else:
            start += ", "
        i+=1
    return start

#Getting Final Output Dictionary
def getAnalysis(output, report_list, priority_list):
    result_list = list(output.keys())
    for i in result_list:
        temp = findString(i)
        output[temp] = output[i]
        del output[i]
    result_list = list(output.keys())
    final_dict = {}
    high_priority_dict = {}
    output_dict = {}
    high_pri = 6
    for i in result_list:
        rep_list = report_list.get(i)
        if rep_list != None:
            priority = priority_list.get(i)
            rep_list['priority'] = priority['priority']
            final_dict[i] = rep_list
            if priority['priority'] < high_pri:
                high_pri = priority['priority']
                high_priority_dict.clear()
                high_priority_dict[i] = rep_list
            elif priority['priority'] == high_pri:
                high_priority_dict[i] = rep_list

    output_dict['start'] = getStart(output)
    temp = output.get(list(high_priority_dict.keys())[0])[0]
    output_dict['suggestion'] = "You should visit a " + str(report_list.get(list(high_priority_dict.keys())[0])[temp][1]) + ", you have chances of " + str(report_list.get(list(high_priority_dict.keys())[0])[temp][0])
    
    for i in list(high_priority_dict.keys()):
        output_dict[str(i)] = generate(i, report_list, output.get(i)[0])
        del output[i]
    for i in list(output.keys()):
        output_dict[str(i)] = generate(i, report_list, output.get(i)[0])
    return output_dict
    # print("X => ", x)
    # print("Y => ", y)
    # print("Final list of all: ", final_dict)
    # print("Highest priority: ", high_priority_dict)

#Output using ChatGPT
def generate(elem, report_list, val):
    # print("\n", elem.upper())
    req_dict = report_list.get(elem)
    # print(req_dict["information"])
    # print(textGenerate("Write a note on " + elem))
    # print(textGenerate("Ill effects of having " + val + " values of " + elem))
    temp = req_dict["remedy_"+val]
    # print("Home remedies to be taken are: ", toString(temp))
    # print(textGenerate("Remedies that can be taken to cure "+ val + " values of " + elem))
    fin = {}
    fin["elem"] = elem
    fin["intro1"] = req_dict["information"]
    fin["intro2"] = textGenerate("Write a note on " + elem)
    fin["intro2"] = fin["intro2"].replace("\n", "")
    fin["effects"] = textGenerate("Ill effects of having " + val + " values of " + elem)
    fin["effects"] = fin["effects"].replace("\n", "")
    fin["rem1"] = "Home remedies to be taken are: " + toString(temp)
    fin["rem2"] = textGenerate("Remedies that can be taken to cure "+ val + " values of " + elem)
    fin["rem2"] = fin["rem2"].replace("\n", "")
    return fin

#ChatGPT
def textGenerate(prompt):
    openai.api_key = OPENAIKEY
    model_engine = "text-davinci-002"
    completion = openai.Completion.create(
        engine = model_engine,
        prompt = prompt,
        max_tokens = 1024,
        n=1,
        stop = None,
        temperature = 0.9,
    )
    response = completion.choices[0].text.lstrip()
    return response

#PDF to Image
def convertpdf2image(file_name):
    pdf = pdfium.PdfDocument(file_name)
    images = []
    for i in range(len(pdf)):
        page = pdf.get_page(i)
        pil_image = page.render_to(
            pdfium.BitmapConv.pil_image,
        )
        file_name = file_name.replace(".", "_")
        output_path = file_name + "_pg" + str(i) + ".jpg"
        images.append(output_path)
        pil_image.save(output_path)
    return images

def main(file_name):
    images = convertpdf2image(file_name)
    for i in range(len(images)):
        table_csv = get_table_csv_results(images[i])
        if table_csv.empty:
            pass
        elif i==0:
            df = table_csv
        elif i==3:
            break
        else:
            df = df.append(table_csv, ignore_index = True)
        os.remove(images[i])
    anomalies = analysis(df)
    output = dict((k.lower(), v) for k, v in anomalies.items()) # {'Eosinophils': ['high', 1], 'MPV (Mean Platelet Volume)': ['high', 0], 'Vitamin B12 level (Serum,CMIA)': ['low', 0]}
    x = getAnalysis(output, report_list, priority_list)

if __name__ == "__main__":
    file_name = sys.argv[1]
    main(file_name)