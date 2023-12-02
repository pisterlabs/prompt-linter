# import os
# from ..postprocessing.sba import SBAPostprocessing
# from ..output_format_nested import slx_output_format as output_format
# from deepdiff import DeepDiff
# import base64
# import json
# import os
# import base64
# from azure.ai.formrecognizer import DocumentAnalysisClient
# from azure.core.credentials import AzureKeyCredential
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
# prefix = "surelogix/sba_dr/"
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
#     # file_name = args[2]
#
#     base64_bytes = base64.b64encode(form)
#     base64_message = base64_bytes.decode("utf8")
#
#     event = {"doc_bytes": base64_message}
#
#     response = lambda_handler(event, None)
#     response = json.loads(response.get('body', '{}'))
#
#     expected = json.loads(response_file)
#
#
#     diff = DeepDiff(expected, response, view="tree")
#
#     diffs = sum([len(diff[x]) for x in diff])
#
#     num_fields = count_fields(expected)
#
#     return num_fields - diffs, num_fields
#
#
# def get_files():
#     s3 = boto3.resource("s3")
#     bucket = s3.Bucket("dml-test-files")
#
#     test_pdfs = []
#     for obj in bucket.objects.filter(Prefix=prefix):
#         if obj.key.endswith(".pdf") and "test" in obj.key:
#             test_pdfs.append(obj)
#
#     test_jsons = []
#
#     for obj in bucket.objects.filter(Prefix=prefix):
#         if obj.key.endswith(".json") and "response" not in obj.key:
#             test_jsons.append(obj)
#
#     args = []
#     for pdf in test_pdfs:
#         for j in test_jsons:
#             if (
#                     pdf.key.split("/")[-1].split("_")[0]
#                     == j.key.split("/")[-1].split("_")[0]
#             ):
#                 args.append(
#                     [
#                         pdf.get()["Body"].read(),
#                         j.get()["Body"].read(),
#                         j.key.split("/")[-1].split("_")[0],
#                     ]
#                 )
#
#     return args
#
#
# def test_sba_dr():
#     args = get_files()
#
#     with Pool() as pool:
#         outs = pool.map(parse_files, args)
#
#     correct = sum([x[0] for x in outs])
#     total = sum([x[1] for x in outs])
#     return correct / total

from .BaseTest import BaseTest

def test_sba_dr():
    files_path = "surelogix/sba_dr"
    test = BaseTest(files_path)
    return test.res
