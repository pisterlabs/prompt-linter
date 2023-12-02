#########################################################################################################################################
## TEST FOR LOCAL DEVELOPMENT
#########################################################################################################################################


# import sys, os, unittest, json, glob, fuzzywuzzy, base64, datetime, re, traceback
#
# sys.path.append(os.getcwd())
#
# import numpy as np
# import pandas as pd
#
# from lambda_functions.surelogix.output_format import slx_output_format
# from lambda_functions.surelogix.surelogix_function_temp import lambda_handler
# from src.preprocess_engine.helper_functions import get_flat_output_from_nested_output_slx
# from src.eval_engine.analytics_functions import get_model_analytics
#
#
# ### Conftest part ###
#
# # 'TEST_FOLDER' should contain three folders:
# # (1) folder with input files
# # (2) a folder named 'output' to dump the results,
# # (3) a folder named 'expected' that contains expected_outputs.json files with correct values
#
#
# folder_path = r"/home/andrey/PycharmProjects/oko-doc-ai/lambda_functions/surelogix/test_docs"
# input_path = os.path.join(folder_path, "input")
# expected_folder = os.path.join(folder_path, "expected")
# output_folder = os.path.join(folder_path, "output_analytics")
#
# # Get the absolute path to the directory containing surelogix_function.py
# script_dir = os.path.dirname(os.path.abspath(__file__))
# surelogix_dir = os.path.join(script_dir, "..", "lambda_functions", "surelogix")
# sys.path.append(surelogix_dir)
#
#
# QA_TEST_MODE = 'DEV'
#
#
# def count_all_folders(folder_path):
#     folder_count = 0
#
#     for root, dirs, files in os.walk(folder_path):
#         folder_count += len(dirs)
#
#     return folder_count
#
#
# def get_datetime_tag():
#     now = datetime.datetime.now()
#     now_date = "0" + str(now.month) if now.month < 10 else str(now.month)
#     now_date += "0" + str(now.day) if now.day < 10 else str(now.day)
#     now_time = "0" + str(now.hour) if now.hour < 10 else str(now.hour)
#     now_time += "0" + str(now.minute) if now.minute < 10 else str(now.minute)
#     return now_date + "-" + now_time
#
# dt_text = get_datetime_tag()
# output_folder_w_test_tag = os.path.join(output_folder, f'QA-{count_all_folders(output_folder)}___' + QA_TEST_MODE + '___at_' + dt_text)
#
# # test-files and test-assets
# pdf_files = glob.glob(input_path + "/*.pdf")
# pdf_names = sorted([pdf_file.split(os.sep)[-1].split(".")[0] for pdf_file in pdf_files])
# all_fields = slx_output_format.keys()
# str_trunc_len = 50
#
#
# ## helper functions
# def get_events_list_from_pdfs():
#     pdf_files = glob.glob(input_path + "/*.pdf")
#     events_list = []
#     for pdf_file in pdf_files:
#         pdf_name = pdf_file.split(os.sep)[-1].split(".")[0]
#         with open(pdf_file, "rb") as pdf_read:
#             form = pdf_read.read()
#         base64_bytes = base64.b64encode(form)
#         base64_message = base64_bytes.decode("utf8")
#         event = {'doc_bytes': base64_message}
#         events_list.append((pdf_name, event))
#
#     return events_list
#
#
# def save_files(data, folder_path, file_name, save_types=['json']):
#     file_path = os.path.join(folder_path, file_name)
#     if 'json' in save_types:
#         with open(file_path + '.json', 'w') as fw:
#             json.dump(data, fw, indent=4)
#
#     if 'csv' in save_types:
#         df = pd.DataFrame(data)
#         df.to_csv(file_path + '.csv', index=False)
#
#
# def get_analytics_summary(output_files, sub_customers_list, file_types_list):
#     model_analytics_summary = {}
#     for sub_customer in sub_customers_list:
#         model_analytics_summary[sub_customer] = {
#             "num_files_tested": 0,
#             "found_fields_ratio": [],
#             "similarity_score_avg": []
#         }
#
#     for file_type in file_types_list.values():
#         model_analytics_summary[file_type] = {
#             "num_files_tested": 0,
#             "found_fields_ratio": [],
#             "similarity_score_avg": []
#         }
#
#     sub_customer = file_type = ""
#     for output_file in output_files:
#         sub_customer_found = False
#         with open(output_file, "r") as output_read:
#             output = json.load(output_read)
#             model_analytics = output['model_analytics']
#         for sub_customer in sub_customers_list:
#             if sub_customer in output_file:
#                 sub_customer_found = True
#                 break
#         if sub_customer_found:
#             model_analytics_summary[sub_customer]["num_files_tested"] += 1
#             model_analytics_summary[sub_customer]["found_fields_ratio"].append(model_analytics.get('found_fields_ratio', 0))
#             model_analytics_summary[sub_customer]['similarity_score_avg'].append(model_analytics.get('avg_similarity_score', 0))
#         for file_type_code in file_types_list:
#             if file_type_code in output_file:
#                 file_type_found = True
#                 break
#         if file_type_found:
#             model_analytics_summary[file_types_list[file_type_code]]["num_files_tested"] += 1
#             model_analytics_summary[file_types_list[file_type_code]]["found_fields_ratio"].append(model_analytics.get('found_fields_ratio', 0))
#             model_analytics_summary[file_types_list[file_type_code]]['similarity_score_avg'].append(model_analytics.get('avg_similarity_score', 0))
#
#     for sub_customer in sub_customers_list:
#         model_analytics_summary[sub_customer]["found_fields_ratio"] = round(np.mean(model_analytics_summary[sub_customer]["found_fields_ratio"]), 2)
#         model_analytics_summary[sub_customer]['similarity_score_avg'] = round(np.mean(model_analytics_summary[sub_customer]['similarity_score_avg']), 2)
#
#     for file_type in file_types_list.values():
#         model_analytics_summary[file_type]["found_fields_ratio"] = round(np.mean(model_analytics_summary[file_type]["found_fields_ratio"]), 2)
#         model_analytics_summary[file_type]['similarity_score_avg'] = round(np.mean(model_analytics_summary[file_type]['similarity_score_avg']), 2)
#
#     return model_analytics_summary
#
#
# def generate_analytics_summary_from_output_files(output_files, sub_customers_list, file_types_list, output_file_name, save_types=['json']):
#     model_analytics_summary = {}
#     for sub_customer in sub_customers_list:
#         for file_type in file_types_list:
#             model_analytics_summary[sub_customer] = {
#                 file_type: {
#                     "num_files_tested": 0,
#                     "found_fields_ratio_req": 0,
#                     "avg_accuracy": 0
#                 }
#             }
#     sub_customer = file_type = ""
#     for output_file in output_files:
#         with open(output_file, "r") as output_read:
#             output = json.load(output_read)
#             model_analytics = output['model_analytics']
#         for sub_customer in sub_customers_list:
#             if sub_customer in output_file:
#                 break
#         for file_type in file_types_list:
#             if file_type in output_file:
#                 break
#         model_analytics_summary[sub_customer][file_type]["num_files_tested"] += 1
#         model_analytics_summary[sub_customer][file_type]["found_fields_ratio_req"] += model_analytics["found_fields_ratio_req"]
#         model_analytics_summary[sub_customer][file_type]["avg_accuracy"] += model_analytics["avg_accuracy"]
#
#     for sub_customer in sub_customers_list:
#         for file_type in file_types_list:
#             num_files_tested = model_analytics_summary[sub_customer][file_type]["num_files_tested"]
#             model_analytics_summary[sub_customer] = {
#                 file_type: {
#                     "num_files_tested": num_files_tested,
#                     "found_fields_ratio_req": round(model_analytics_summary[sub_customer][file_type]["found_fields_ratio_req"] / num_files_tested,
#                                                     2) if num_files_tested else None,
#                     "avg_accuracy": round(model_analytics_summary[sub_customer][file_type]["avg_accuracy"] / num_files_tested, 1) if num_files_tested else None,
#                 }
#             }
#
#
# def generate_output_from_pdf_name(pdf_name, output_folder, expected_folder=expected_folder, flat_output=True):
#     events_list = get_events_list_from_pdfs()
#     for name, event in events_list:
#         if name == pdf_name:
#             # init from pdf_name:
#             lambda_result = lambda_handler(event, None)  # type: ignore
#
#             print('lambda_result', lambda_result)
#
#             lambda_result_body = json.loads(lambda_result.get('body'))['order_list']
#             customer_info = lambda_result.get('customer_info')
#             model_info = lambda_result.get('model_info')
#             if not os.path.exists(output_folder):
#                 os.makedirs(output_folder)
#             output_file = os.path.join(output_folder, pdf_name + ".json")
#             expected_file = os.path.join(expected_folder, pdf_name + ".json")
#
#             if flat_output:
#                 lambda_result_body_flat = get_flat_output_from_nested_output_slx(lambda_result_body)
#
#             model_analytics = {}
#             try:
#                 with open(expected_file, "r") as expected_read:
#                     output_expected = json.load(expected_read)
#                     if 'body' in output_expected:
#                         output_expected = output_expected['body']
#
#                 model_analytics = get_model_analytics(output_format=slx_output_format, output_expected=output_expected, output_result=lambda_result_body_flat)
#
#                 # TODO-P1----------
#                 if not output_expected['goods.pieces']: output_expected['goods.pieces'] = []
#                 for ind, piece in enumerate(output_expected['goods.pieces']):
#                     for key, val in piece.items():
#                         if key.lower() == 'description':
#                             continue
#                         elif key.lower() == 'weight':
#                             wei_exp = output_expected['goods.pieces'][ind][key].split('.')[0]
#                             wei_res = lambda_result_body['goods']['pieces'][ind][key].split('.')[0]
#                             if wei_exp != wei_res:
#                                 piece_not_match = {
#                                     "field": f"goods.pieces.{ind}.{key}",
#                                     "result": lambda_result_body['goods']['pieces'][ind][key],
#                                     "expected": output_expected['goods.pieces'][ind][key]
#                                 }
#                                 model_analytics["list_not_perfect_fields"].append(piece_not_match)
#                         elif output_expected['goods.pieces'][ind][key] != lambda_result_body['goods']['pieces'][ind][key]:
#                             piece_not_match = {
#                                 "field": f"goods.pieces.{ind}.{key}",
#                                 "result": lambda_result_body['goods']['pieces'][ind][key],
#                                 "expected": output_expected['goods.pieces'][ind][key]
#                             }
#                             model_analytics["list_not_perfect_fields"].append(piece_not_match)
#             except Exception as e:
#                 traceback.print_exc()
#                 print(f" ~~~ ERROR ~~~ '{e}!")
#
#                 # ----------
#             try:
#                 model_performance = {
#                     'found_fields_ratio': model_analytics['found_fields_ratio'],
#                     'avg_accuracy': model_analytics['avg_similarity_score'],
#                 }
#             except:
#                 model_performance = {}
#
#             output_to_save = {
#                 'model_performance': model_performance,
#                 'body': lambda_result_body_flat,
#                 'model_analytics': model_analytics,
#                 'customer_info': customer_info,
#                 'model_info': model_info
#             }
#             with open(output_file, "w") as fw:
#                 json.dump(output_to_save, fw, indent=4)
#             print(f"{pdf_name} is saved")
#
#
# def generate_outputs_from_local_files(pdf_names, output_folder):
#     for pdf_name in pdf_names:
#         generate_output_from_pdf_name(pdf_name=pdf_name, output_folder=output_folder)
#
#
# def make_string_clean(original_string):
#     # Use reg exp to remove unwanted characters (spaces, hyphens, and commas) & truncate
#     return re.sub(r'[ ,\-]', '', original_string.lower())[:str_trunc_len]
#
#
# ###
# class TestDevModel(unittest.TestCase):
#     def __init__(self, methodName: str = 'runTest') -> None:
#         super().__init__(methodName)
#
#         """
#         Different "QA result-vs-Expected Tests" in different stages: "Dev", "Deploy"
#         """
#         self.QA_TEST_MODE = QA_TEST_MODE
#         self.MODEL_NAME = 'SLX-V7'
#         # set during testing
#         self.generate_new_outputs = True
#         self.flat_output = True
#         self.test_fields = [key for key, val in slx_output_format.items() if '#req' in val and '#met' in val]
#         self.num_test_files = 20
#         self.test_files = {
#             'sub_customers': ['efw', 'alg', 'omni', 'allstate', 'icat', 'pegasus', 'unwanted'],
#             'file_types': {
#                 'da': 'deliver-alert',
#                 'am': 'alert-manifest'
#             }
#         }
#         self.dt_text = get_datetime_tag()
#         self.output_folder_w_test_tag = output_folder_w_test_tag
#         self.output_files_prefix = os.path.join(output_folder, f'{self.MODEL_NAME}___QA_TESTS_on_{self.QA_TEST_MODE}___dt_{self.dt_text}')
#         self.acceptable_found_fields_ratio = 95
#         self.acceptable_similarity_score = 95
#
#     def test_compare_expected_to_output(self):
#         found_fields_ratios = []
#         avg_similarity_score = []
#         num_files_detected_unwanted = 0
#         for pdf_ind, pdf_name in enumerate(pdf_names[:self.num_test_files]):
#             print(' ')
#             # either Generate-Outputs OR Read-Latest-Outputs
#             if self.generate_new_outputs:
#                 generate_output_from_pdf_name(pdf_name=pdf_name, output_folder=self.output_folder_w_test_tag)
#                 output_file = self.output_folder_w_test_tag + f'/{pdf_name}.json'
#             else:
#                 output_folder_latest = max([folder for folder in glob.glob(output_folder + '/*') if os.path.isdir(folder)], key=os.path.getctime)
#                 output_file = output_folder_latest + f'/{pdf_name}.json'
#             try:
#                 with open(output_file, 'r') as output_read:
#                     output = json.load(output_read)
#                     output_data = output['body']
#                     model_analytics = output['model_analytics']
#             except Exception as e:
#                 print(f' ~~~ ERROR ~~~ {e}!')
#                 continue
#
#             try:
#                 print(f' ========= Test #: {pdf_ind + 1} ====== {pdf_name.replace(" ", "")} ========= ')
#                 # found_fields_ratio analytics:
#                 if model_analytics.get('found_fields_ratio', 0) < self.acceptable_found_fields_ratio:
#                     print('--> "found_fields_ratio": ', model_analytics['found_fields_ratio'])
#                 else:
#                     print('--> SUCCESS "found_fields_ratio": ', model_analytics['found_fields_ratio'])
#                 found_fields_ratios.append(model_analytics['found_fields_ratio'])
#
#                 # avg_similarity_score analytics:
#                 if model_analytics['avg_similarity_score'] < self.acceptable_found_fields_ratio:
#                     print('--> "avg_similarity_score": ', model_analytics['avg_similarity_score'])
#                 else:
#                     print('--> SUCCESS "avg_similarity_score": ', model_analytics['avg_similarity_score'])
#                 avg_similarity_score.append(model_analytics['avg_similarity_score'])
#
#                 print(
#                     f" ==================== Fields Found Ratio = {model_analytics['found_fields_ratio']} === Avg Similarity Score = {model_analytics['avg_similarity_score']} ==================== ")
#                 print(' ')
#                 print(' ')
#             except Exception as e:
#                 num_files_detected_unwanted += 1
#                 print(f' ~~~ ERROR ~~~ {e}!')
#
#         # save anlaytics summary:
#         output_folder_latest = max([folder for folder in glob.glob(output_folder + '/*') if os.path.isdir(folder)], key=os.path.getctime)
#         output_files = glob.glob(output_folder_latest + '/*.json')
#         model_analytics_summary = get_analytics_summary(output_files=output_files,
#                                                         sub_customers_list=self.test_files['sub_customers'],
#                                                         file_types_list=self.test_files['file_types'])
#         model_analytics_summary['unwanted'] = {
#             'num_files_tested': num_files_detected_unwanted,
#             'num_files_detected_unwanted': num_files_detected_unwanted,
#         }
#         save_files(data=model_analytics_summary, folder_path=output_folder, file_name=f'{self.output_files_prefix}___analytics_summary')


# ###########################################################################################################################################
# ## TEST FOR GIT REPO
# ###########################################################################################################################################
# from deepdiff import DeepDiff
# import json
# import os
# import base64
# import openai
# from multiprocessing import Pool
# import warnings
# import asyncio
# import boto3
# from ..surelogix_function_temp import lambda_handler
#
# if os.getenv("ENVIRONMENT") == "dev":
#     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
# warnings.filterwarnings("ignore")
#
# azure_endpoint = os.environ["AZURE_ENDPOINT"]
# azure_key = os.environ["AZURE_KEY"]
# openai.api_key = os.environ["OPENAI_API_KEY"]
#
#
# # create a function that counts how many fields a dictionary has. If a field is a dictionary, count its fields too.
# def count_fields(d):
#     if type(d) is dict:
#         return sum([count_fields(v) for v in d.values()])
#     elif type(d) is list:
#         return sum([count_fields(v) for v in d])
#     else:
#         return 1
#
#
# def parse_files(args):
#     response_file = args[1]
#     form = args[0]
#     file_name = args[2]
#
#     base64_bytes = base64.b64encode(form)
#     base64_message = base64_bytes.decode("utf8")
#
#     event = {"doc_bytes": base64_message}
#
#     response = lambda_handler(event, None)
#     order_data = response.get('order_list')
#
#     expected = json.loads(response_file)
#
#     diff = DeepDiff(expected, order_data, view="tree")
#
#     diffs = sum([len(diff[x]) for x in diff])
#     num_fields = count_fields(expected)
#
#     return num_fields - diffs, num_fields
#
#
# def get_files():
#     s3 = boto3.resource("s3")
#     bucket = s3.Bucket("dml-test-files")
#     test_pdfs = []
#     for obj in bucket.objects.filter(Prefix="surelogix/sba_da/"):
#         if obj.key.endswith(".pdf"):
#             test_pdfs.append(obj)
#
#     test_jsons = []
#
#     for obj in bucket.objects.filter(Prefix="surelogix/sba_da/"):
#         if obj.key.endswith(".json"):
#             test_jsons.append(obj)
#
#     args = []
#     for pdf in test_pdfs:
#         for j in test_jsons:
#             if (
#                     pdf.key.split("/")[-1].split(".")[0]
#                     == j.key.split("/")[-1].split(".")[0]
#             ):
#                 args.append(
#                     [
#                         pdf.get()["Body"].read(),
#                         j.get()["Body"].read(),
#                         j.key.split("/")[-1].split(".")[0],
#                     ]
#                 )
#
#     return args
#
#
# def test_sba_da():
#     args = get_files()
#
#     with Pool() as pool:
#         outs = pool.map(parse_files, args)
#
#     correct = sum([x[0] for x in outs])
#     total = sum([x[1] for x in outs])
#
#     res = correct / total if total else 0
#
#     print('Test Result: ', res)
#
#     return res

from .BaseTest import BaseTest

def test_sba_da():
    files_path = "surelogix/sba_da/"
    test = BaseTest(files_path)
    return test.res
