import csv
from math import isnan
import ast, io, random, sys, os
import asyncio
from typing import Optional, Union

from openai import OpenAIError
from fastapi import APIRouter, status, UploadFile, File
from fastapi.responses import JSONResponse

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from mitaskem.src.connect import construct_data_card, dataset_header_document_dkg, construct_model_card, profile_matrix, get_dataset_type, process_data
from mitaskem.src.response_types import MatrixDataCard, TabularDataCard, ModelCard

router = APIRouter()

@router.post("/get_data_card", tags=["Data-and-model-cards"], response_model=Union[MatrixDataCard, TabularDataCard])
async def get_data_card(gpt_key: str, csv_file: UploadFile = File(...), doc_file: UploadFile = File(...), smart: Optional[bool] = False):
    """
           Smart run provides better results but may result in slow response times as a consequence of extra GPT calls.
    """
    files = [csv_file.read(), doc_file.read()]
    _csv, doc = await asyncio.gather(*files)
    _csv = _csv.decode().strip()
    doc = doc.decode().strip()

    # TODO handle inputs that are too long to fit in the context window
    if len(_csv) == 0:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content="Empty CSV file")
    if len(doc) == 0:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content="Empty document file")

    lines = _csv.splitlines()
    print('csv_file head (first 10 lines):')
    for l in lines[:10]:
        print(l)

    doc_lines = doc.splitlines()
    print('doc_file head (first 5 lines):')
    for l in doc_lines[:5]:
        print(l)

    csv_reader = csv.reader(io.StringIO(_csv), dialect=csv.Sniffer().sniff(_csv.splitlines()[-1]))

    header = next(csv_reader)  # can determine type from the header row
    data_type = get_dataset_type(header)
    if data_type == 'header-0':
        schema = header
        profiler = dataset_header_document_dkg
    elif data_type == 'no-header':
        # Probably best not to support this; the code path is poorly tested, and it's not clear what the expected behavior is.
        # Either way, this should never come up in the Evaluation.
        #schema = None
        #profiler = dataset_header_dkg
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content="Invalid CSV file; no header found.")
    elif data_type == 'matrix':
        schema = None
        profiler = profile_matrix
    else:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content="Invalid CSV file; could not determine data type")

    data = [header]
    data.extend(csv_reader)  # make sure header is included in data
    data = process_data(data)

    calls = [
        construct_data_card(data_doc=doc, dataset_name=csv_file.filename, doc_name=doc_file.filename, dataset_type=data_type, gpt_key=gpt_key),
        profiler(data=data, doc=doc, dataset_name=csv_file.filename, doc_name=doc_file.filename, gpt_key=gpt_key, smart=smart)
    ]

    try:
        results = await asyncio.gather(*calls)
    except OpenAIError as err:
        if "maximum context" in str(err):
            return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content="Input too long. Please reduce the size of your input.")
        else:
            return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=f"OpenAI connection error: {err}")

    for s, success in results:
        if not success:
            return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=s)

    data_card = ast.literal_eval(results[0][0])
    data_profiling = ast.literal_eval(results[1][0])
    if 'DATA_PROFILING_RESULT' in data_card:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content='DATA_PROFILING_RESULT cannot be a requested field in the data card.')

    if data_type == 'header-0':
        data_card['SCHEMA'] = schema
        # get a random sample of a row from the csv
        data_card['EXAMPLES'] = {k.strip(): v for k, v in zip(schema, random.sample(list(data[1:]), 1)[0])}
        data_card['DATA_PROFILING_RESULT'] = data_profiling
    elif data_type == 'no-header':
        if 'SCHEMA' not in data_card:
            return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content='SCHEMA not found in data card')
        schema = [s.strip() for s in data_card['SCHEMA'].split(',')]
        schema = [s[1:] if s.startswith('[') else s for s in schema]
        schema = [s[:-1] if s.endswith(']') else s for s in schema]
        aligned_data_profiling = {}
        for k, v in data_profiling.items():
            k = int(k)
            k = schema[k]
            aligned_data_profiling[k] = v
        data_card['DATA_PROFILING_RESULT'] = aligned_data_profiling
        data_card['EXAMPLES'] = {k.strip(): v for k, v in zip(schema, random.sample(list(data), 1)[0])}
    elif data_type == 'matrix':
        data_card['DATA_PROFILING_RESULT'] = data_profiling
        data_card['EXAMPLES'] = random.sample(data, 1)[0]
    else:
        raise Exception('Invalid data type')

    def _fill_nan(ex):
        if isinstance(ex, dict):
            for k, v in ex.items():
                ex[k] = _fill_nan(v)
        elif isinstance(ex, list):
            for i in range(len(ex)):
                ex[i] = _fill_nan(ex[i])
        elif isinstance(ex, float) and isnan(ex):
            ex = None
        return ex

    data_card['EXAMPLES'] = _fill_nan(data_card['EXAMPLES'])

    data_card['DATASET_TYPE'] = "matrix" if data_type == 'matrix' else "tabular"

    print(data_card)
    return data_card

@router.post("/get_model_card", tags=["Data-and-model-cards"], response_model=ModelCard)
async def get_model_card(gpt_key: str, text_file: UploadFile = File(...), code_file: UploadFile = File(...)):

    files = [text_file.read(), code_file.read()]
    text, code = await asyncio.gather(*files)

    # process model text
    text_string = text.decode()

    # process code
    code_string = code.decode()

    try:
        res, success = await construct_model_card(text=text_string, code=code_string, gpt_key=gpt_key)
    except OpenAIError as err:
        if "maximum context" in str(err):
            return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content="Input too long. Please reduce the size of your input.")
        else:
            return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=f"OpenAI connection error: {err}")

    if not success:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=res)
    model_card = ast.literal_eval(res)
    return model_card
