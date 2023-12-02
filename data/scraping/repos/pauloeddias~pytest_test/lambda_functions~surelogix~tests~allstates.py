import os
from ..postprocessing.alg import ALGPostprocessing
from ..output_format_nested import slx_output_format as output_format
from deepdiff import DeepDiff
import base64
import json
import os
import base64
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
import openai
from multiprocessing import Pool
import warnings
import asyncio
import boto3
from ..surelogix_function_temp import lambda_handler

if os.getenv("ENVIRONMENT") == "dev":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
warnings.filterwarnings("ignore")

azure_endpoint = os.environ["AZURE_ENDPOINT"]
azure_key = os.environ["AZURE_KEY"]
openai.api_key = os.environ["OPENAI_API_KEY"]


# create a function that counts how many fields a dictionary has. If a field is a dictionary, count its fields too.
def count_fields(d):
    if type(d) is dict:
        return sum([count_fields(v) for v in d.values()])
    elif type(d) is list:
        return sum([count_fields(v) for v in d])
    else:
        return 1


def parse_files(args):
    response_file = args[1]
    form = args[0]
    # file_name = args[2]

    base64_bytes = base64.b64encode(form)
    base64_message = base64_bytes.decode("utf8")

    event = {"doc_bytes": base64_message}

    response = lambda_handler(event, None)
    response = json.loads(response.get('body','{}'))

    # if os.getenv("ENVIRONMENT") == "dev":
    #     # response = json.loads(response)
    #     json.dump(response, open("lambda_functions/surelogix/files/ALG/" + file_name + "_response.json", "w"), indent=1)

    expected = json.loads(response_file)

    diff = DeepDiff(expected, response, view="tree")

    diffs = sum([len(diff[x]) for x in diff])
    num_fields = count_fields(expected)

    return num_fields - diffs, num_fields


def get_files():
    s3 = boto3.resource("s3")
    bucket = s3.Bucket("dml-test-files")

    test_pdfs = []
    for obj in bucket.objects.filter(Prefix="surelogix/allstates/hawb/"):
        if obj.key.endswith(".pdf"):
            test_pdfs.append(obj)

    test_jsons = []

    for obj in bucket.objects.filter(Prefix="surelogix/allstates/hawb/"):
        if obj.key.endswith(".json"):
            test_jsons.append(obj)

    args = []
    for pdf in test_pdfs:
        for j in test_jsons:
            if (
                    pdf.key.split("/")[-1].split(".")[0]
                    == j.key.split("/")[-1].split(".")[0]
            ):
                args.append(
                    [
                        pdf.get()["Body"].read(),
                        j.get()["Body"].read(),
                        j.key.split("/")[-1].split(".")[0],
                    ]
                )

    return args

def get_files_local():
    path = 'lambda_functions/surelogix/files/ALG/'
    files = os.listdir(path)
    test_pdfs = []
    for obj in files:
        if obj.endswith(".pdf") and "test" in obj:
            test_pdfs.append(obj)

    test_jsons = []
    for obj in files:
        if obj.endswith(".json") and "test" in obj:
            test_jsons.append(obj)

    args = []
    for pdf in test_pdfs:
        for j in test_jsons:
            if (
                    pdf.split("/")[-1].split(".")[0]
                    == j.split("/")[-1].split(".")[0]
            ):
                with open(path + pdf, "rb") as file:
                    pd = file.read()
                with open(path + j, "rb") as file:
                    js = file.read()
                args.append(
                    [
                        pd,
                        js,
                        j.split("/")[-1].split(".")[0],
                    ]
                )

    return args

def test_allstates_hawb():
    args = get_files()

    with Pool() as pool:
        outs = pool.map(parse_files, args)

    correct = sum([x[0] for x in outs])
    total = sum([x[1] for x in outs])

    res = correct / total
    print('Test Result: ', res)

    return res


