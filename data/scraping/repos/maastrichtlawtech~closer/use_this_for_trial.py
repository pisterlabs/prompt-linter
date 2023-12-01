import json
import os
import sys

import nltk
import pandas as pd
import xlsxwriter
from dotenv import load_dotenv

from legal_openai.openai_tasks import OpenaiTask

load_dotenv()
tagme_api_key = os.getenv("GCUBE_TOKEN")
openai_api_key = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Initialise dependencies
tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()

# Recognise entities in the text
# By iterating through all articles in the articles folder
# Or parse it from the html link
'''
html_link = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32019R0947&from=EN"
legal_text = get_text_from(html_link)

# Read schema for data storage
with open('./input/schema.json', 'r') as f:
    json_schema = json.load(f)

# Add more structure to the schema
json_schema["document_data"] = {}
json_schema["document_data"]["uri"] = html_link
json_schema["document_data"]["sections"] = {}
json_schema["document_data"]["sections"]["chapters"] = []

# Process the text
processed_legal_text = process_text(legal_text)

# Keep track of articles. Temp setup until article number, name are extracted
# in helper functions.
article_count = 0
articles = []
'''
# Initialise entity recogniser object
# entity = EntityRecognizer()
# Output directories
output_entity_base_dir = './output/entity/'
output_quantity_base_dir = './output/quantity/'
output_reference_base_dir = './output/reference/'
output_deontic_modality_base_dir = './output/deontic_modality/'
output_exceptions_base_dir = './output/exceptions/'
output_scope_base_dir = './output/scope/'
output_definitions_base_dir = './output/definitions/'
output_if_then_base_dir = './output/if_then/'

# Prompt directories
prompt_path_human = './prompts/human_annotation_guidelines/'
prompt_path_role_playing = './prompts/role_playing/'
prompt_path_cot = './prompts/cot/'
prompt_path_normal_prompt = './prompts/normal_prompts/'

# Excel file names
spacy_excel_file = 'spacy'
eurovoc_excel_file = 'eurovoc'
wikipedia_excel_file = 'wikipedia'
refined_excel_file = 'refined'
openai_excel_file = 'openai'
quantulum_excel_file = 'quantulum'

# Column names
definitions_column = ['article', 'prompt_method', 'definition_term', 'definition_text',
                      'relationship', 'reference_text', 'reference_relationship']
deontic_modality_column = ['article', 'prompt_method', 'atomic_statement/text', 'class/type', 'action',
                           'active_role/norm_addresse', 'passive_role/beneficiary']
exceptions_column = ['article', 'prompt_method', 'exception']
if_then_column = ['article', 'prompt_method', 'if_statement',
                  'then_statement', 'condition_type']
quantities_column = ['article', 'prompt_method', 'quantity', 'unit']
references_column = ['article', 'prompt_method', 'reference']
scope_column = ['article', 'prompt_method', 'scope', 'scope_type']

# Prompt path list
prompt_paths = [prompt_path_human, prompt_path_role_playing,
                prompt_path_cot, prompt_path_normal_prompt]


# Write output to excel sheets
def write_excelsheet(output, file, article_split, task_type='entity'):
    if output is not None:
        with xlsxwriter.Workbook(file) as workbook:
            worksheet = workbook.add_worksheet(article_split)
            row = 0
            col = 0
            if task_type == 'entity':
                print(f'{file}')
                worksheet.write(row, col, 'Entity')
                worksheet.write(row, col + 1, 'URI')
                for key in output.keys():
                    row += 1
                    worksheet.write(row, col, key)
                    worksheet.write(row, col + 1, output[key])
            elif task_type == 'quantity':
                worksheet.write(row, col, task_type)
                for key in output:
                    row += 1
                    worksheet.write(row, col, key)
            elif type(output) is list:
                worksheet.write(row, col, task_type)
                for i in output:
                    if type(i) is dict:
                        for key in i.keys():
                            row += 1
                            worksheet.write(row, col, key)
                            worksheet.write(row, col + 1, i[key])
                    elif type(i) is str:
                        row += 1
                        worksheet.write(row, col, i)
            elif task_type == 'deontic_logic' and type(output) is dict:
                worksheet.write(row, col, task_type)
                for key in output.keys():
                    row += 1
                    worksheet.write(row, col, key)
                    worksheet.write(row, col + 1, output[key])
            else:
                worksheet.write(row, col, task_type)
                for key in output.keys():
                    row += 1
                    worksheet.write(row, col, key)
                    worksheet.write(row, col + 1, output[key])


# Process data
def data_process(article=None, task=None, data=None, prompt_type=None):
    try:
        data = {k.strip(' \t\n\r'): v for k, v in data.items()}
    except AttributeError:
        try:
            with open(f'./output/definitions/{article}_{task}.json', 'w') as f:
                json.dump(data, f, indent=4)
                print(f'Exception in {article}')
        except Exception as e:
            print(f'Exception in {article}')
            print(e)
        if task == 'definition':
            temp_df = pd.DataFrame(columns=definitions_column)
            temp_df.loc[len(temp_df)] = [article, prompt_type, 'In the JSON file',
                                         'NA', 'NA', 'NA', 'NA']
            return temp_df
        elif task == 'deontic_modality':
            temp_df = pd.DataFrame(columns=deontic_modality_column)
            temp_df.loc[len(temp_df)] = [article, prompt_type, 'In the JSON file',
                                         'NA', 'NA', 'NA', 'NA']
            return temp_df
        elif task == 'exception':
            temp_df = pd.DataFrame(columns=exceptions_column)
            temp_df.loc[len(temp_df)] = [article, prompt_type, 'In the JSON file']
            return temp_df
        elif task == 'if_then':
            temp_df = pd.DataFrame(columns=if_then_column)
            temp_df.loc[len(temp_df)] = [article, prompt_type, 'In the JSON file',
                                         'NA']
            return temp_df
        elif task == 'quantity':
            temp_df = pd.DataFrame(columns=quantities_column)
            temp_df.loc[len(temp_df)] = [article, prompt_type, 'In the JSON file',
                                         'NA']
            return temp_df
        elif task == 'reference':
            temp_df = pd.DataFrame(columns=references_column)
            temp_df.loc[len(temp_df)] = [article, prompt_type, 'In the JSON file']
            return temp_df
        elif task == 'scope':
            temp_df = pd.DataFrame(columns=scope_column)
            temp_df.loc[len(temp_df)] = [article, prompt_type, 'In the JSON file',
                                         'NA']
            return temp_df

    if task == 'definition':
        definitions_df = pd.DataFrame(columns=definitions_column)
        for key in data.keys():
            if key == 'definition':
                for i in data[key]:
                    try:
                        if 'reference' in i and i['reference'] is not None:
                            temp_list = [article, prompt_type, i['definition_term'],
                                         i['definition_text'], i['relationship'],
                                         i['reference']['text'],
                                         i['reference']['relationship']]
                            definitions_df.loc[len(definitions_df)] = temp_list
                        else:
                            temp_list = [article, prompt_type, i['definition_term'],
                                         i['definition_text'], i['relationship'],
                                         'NA', 'NA']
                            definitions_df.loc[len(definitions_df)] = temp_list
                    except KeyError:
                        with open(f'./output/definitions/{article}_{task}.json', 'w') as f:
                            json.dump(data, f, indent=4)
                            print(f'Exception in {article}')
                        temp_list = [article, prompt_type, 'In the JSON file',
                                     'NA', 'NA', 'NA', 'NA']
                        definitions_df.loc[len(definitions_df)] = temp_list
        return definitions_df
    if task == 'deontic_modality':
        deontic_modality_df = pd.DataFrame(columns=deontic_modality_column)
        for key in data.keys():
            if key == 'deontic_modality':
                for i in data[key]:
                    try:
                        if prompt_type == 'normal_prompts':
                            deontic_modality_df.loc[len(deontic_modality_df)] = [article,
                                                                                 prompt_type,
                                                                                 i['atomic_statement'],
                                                                                 i['type'],
                                                                                 i['action'],
                                                                                 i['active_role'],
                                                                                 i['passive_role']]
                        else:
                            deontic_modality_df.loc[len(deontic_modality_df)] = [article,
                                                                                 prompt_type,
                                                                                 i['text'],
                                                                                 i['class'],
                                                                                 'NA',
                                                                                 i['norm_addressee'],
                                                                                 i['beneficiary']]
                    except KeyError:
                        with open(f'./output/deontic_modality/{article}_{task}.json', 'w') as f:
                            json.dump(data, f, indent=4)
                        deontic_modality_df.loc[len(deontic_modality_df)] = [article,
                                                                             prompt_type,
                                                                             'In the JSON file',
                                                                             'NA', 'NA', 'NA', 'NA']
        return deontic_modality_df
    if task == 'exceptions':
        exceptions_df = pd.DataFrame(columns=exceptions_column)
        for key in data.keys():
            if key == 'exceptions':
                for i in data[key]:
                    exceptions_df.loc[len(exceptions_df)] = [article,
                                                             prompt_type,
                                                             i['exception']]
        return exceptions_df
    if task == 'if_then':
        if_then_df = pd.DataFrame(columns=if_then_column)
        for key in data.keys():
            if key == 'if-then-statements':
                for i in data[key]:
                    if_then_df.loc[len(if_then_df)] = [article,
                                                       prompt_type,
                                                       i['if'],
                                                       i['then'],
                                                       i['condition_type']]
        return if_then_df
    if task == 'quantity':
        quantities_df = pd.DataFrame(columns=quantities_column)
        for key in data.keys():
            if key == 'quantities':
                for i in data[key]:
                    if 'unit' not in i or i['unit'] is None:
                        temp_list = [article, prompt_type, i['value'], 'None']
                    else:
                        temp_list = [article, prompt_type, i['value'], i['unit']]
                    quantities_df.loc[len(quantities_df)] = temp_list
        return quantities_df
    if task == 'references':
        references_df = pd.DataFrame(columns=references_column)
        for key in data.keys():
            if key == 'reference':
                for i in data[key]:
                    references_df.loc[len(references_df)] = [article,
                                                             prompt_type,
                                                             i['text']]
        return references_df

    if task == 'scope':
        scope_df = pd.DataFrame(columns=scope_column)
        for key in data.keys():
            if key == 'scope':
                for i in data[key]:
                    if 'scope_type' not in i:
                        temp_list = [article, prompt_type, i['text'], 'None']
                    else:
                        temp_list = [article, prompt_type, i['text'], i['scope_type']]
                    scope_df.loc[len(scope_df)] = temp_list
        return scope_df


# Method to run definition recognition will prompts for a given article
def execute_tasks(article=None, openai_obj=None, task=None, prompt=None):
    if article is None:
        print("No article provided, exiting")
        sys.exit(1)
    elif article == 'article_omission.txt':
        with open('./prompt_sdg.txt', 'r') as f:
            prompt = f.read()
        temp_dict = openai_obj.execute_task(article='article_omission', prompt=prompt)
        with open('./synthetic_data_omission.json', 'w') as f:
            json.dump(temp_dict, f, indent=4)
    else:
        processed_data_final = pd.DataFrame()
        processed_data = pd.DataFrame()
        if prompt is None:
            for prompt_path in prompt_paths:
                print(f"Processing with {prompt_path}")
                if task == 'definition':
                    file_name = 'definitions.txt'
                elif task == 'deontic_modality':
                    file_name = 'deontic_modality.txt'
                elif task == 'exceptions':
                    file_name = 'exceptions.txt'
                elif task == 'references':
                    file_name = 'references.txt'
                elif task == 'if_then':
                    file_name = 'if_then.txt'
                elif task == 'quantity':
                    file_name = 'quantity.txt'
                elif task == 'scope':
                    file_name = 'scope.txt'
                with open(prompt_path + file_name, 'r', encoding='utf-8') as f:
                    prompt = f.read()
                temp_dict = openai_obj.execute_task(article=article_split, prompt=prompt)
                if temp_dict is not None:
                    processed_data = data_process(article=article,
                                                  task=task,
                                                  data=temp_dict,
                                                  prompt_type=prompt_path.split('/')[2].strip())
                processed_data_final = pd.concat([processed_data_final, processed_data],
                                                 ignore_index=True)
                print(f"Output of processed data {processed_data_final}")
        else:
            if prompt is not None:
                temp_dict = openai_obj.execute_task(article=article_split, prompt=prompt)
                if temp_dict is not None:
                    processed_data_final = data_process(article=article,
                                                        task=task,
                                                        data=temp_dict,
                                                        prompt_type='manual')
            else:
                print("No prompt provided, exiting")
                sys.exit(1)
        return processed_data_final


# Provide path from which files are to be read
# Not as sub directories but just text files
_path = './input/test_provisions/'
for article in os.listdir(_path):
    if article.endswith('.txt'):
        article_split = article.split('.txt')[0]
        # Change index value to False if you don't want to index the text
        openai_obj = OpenaiTask(path=_path, temperature=0, use_index=True)
        '''
        # Definition recognition with openai
        print(f"Processing {article} with openai for definitions")
        if not os.path.exists(output_definitions_base_dir + str(article_split) +
                              '_definitions.csv'):
            processed_data = execute_tasks(article=article_split, task='definition',
                                           openai_obj=openai_obj)
            if processed_data is not None:
                processed_data.to_csv(output_definitions_base_dir + str(article_split) +
                                      '_definitions.csv', sep=',', encoding='utf-8',
                                      index=False)
        '''
        # Deontic logic recognition with openai
        print(f"Processing {article} with openai for deontic modality")
        if not os.path.exists(output_deontic_modality_base_dir + str(article_split) +
                              '_deontic_modality.csv'):
            # if index is set as true
            processed_data = execute_tasks(article=article_split,
                                           task='deontic_modality',
                                           openai_obj=openai_obj)
            '''
            # if index is set as false
            with open('./prompts/manual_prompt.txt', 'r', encoding='utf-8') as f:
                deontic_modality_prompt = f.read()

            processed_data = execute_tasks(article=article_split,
                                           task='deontic_modality',
                                           openai_obj=openai_obj,
                                            prompt=deontic_modality_prompt)
            '''
            if processed_data is not None:
                processed_data.to_csv(output_deontic_modality_base_dir +
                                      str(article_split) +
                                      '_deontic_modality.csv', index=False)
        '''
        # Exceptions recognition with openai
        print(f"Processing {article} with openai for exceptions")
        if not os.path.exists(output_exceptions_base_dir + str(article_split) +
                              '_exceptions.csv'):
            processed_data = execute_tasks(article=article_split, task='exceptions',
                                           openai_obj=openai_obj)
            if processed_data is not None:
                processed_data.to_csv(output_exceptions_base_dir + str(article_split) +
                                      '_exceptions.csv', index=False)
        # If then recognition with openai
        print(f"Processing {article} with openai for if then")
        if not os.path.exists(output_if_then_base_dir + str(article_split) +
                              '_if_then.csv'):
            processed_data = execute_tasks(article=article_split, task='if_then',
                                           openai_obj=openai_obj)
            if processed_data is not None:
                processed_data.to_csv(output_if_then_base_dir + str(article_split) +
                                      '_if_then.csv', index=False)
        # Quantity recognition with openai
        print(f"Processing {article} with openai for quantity")
        if not os.path.exists(output_quantity_base_dir + str(article_split) +
                              '_quantity.csv'):
            processed_data = execute_tasks(article=article_split, task='quantity',
                                           openai_obj=openai_obj)
            if processed_data is not None:
                processed_data.to_csv(output_quantity_base_dir+ str(article_split) +
                                      '_quantity.csv', index=False)
        # References recognition with openai
        print(f"Processing {article} with openai for references")
        if not os.path.exists(output_reference_base_dir + str(article_split) +
                              '_references.csv'):
            processed_data = execute_tasks(article=article_split, task='references',
                                           openai_obj=openai_obj)
            if processed_data is not None:
                processed_data.to_csv(output_reference_base_dir + str(article_split) +
                                      '_references.csv', index=False)
        # Scope recognition with openai
        print(f"Processing {article} with openai for scope")
        if not os.path.exists(output_scope_base_dir + str(article_split) +
                              '_scope.csv'):
            processed_data = execute_tasks(article=article_split, task='scope',
                                           openai_obj=openai_obj)
            if processed_data is not None:
                processed_data.to_csv(output_scope_base_dir + str(article_split) +
                                      '_scope.csv', index=False)
        '''
