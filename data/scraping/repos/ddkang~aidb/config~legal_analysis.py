from aidb.config.config_types import AIDBListType
from aidb.inference.examples.detectron_local_ocr import DetectronLocalOCR
from aidb.inference.examples.openai_inference_service import OpenAIText


DB_URL = 'sqlite+aiosqlite://'
DB_NAME = 'aidb_test_legal.sqlite'

ocr = DetectronLocalOCR(
  name="ocr",
  model_path="tests/data/model_final.pth")

openai_gpt = OpenAIText(
  name="openai_gpt",
  token=None, # leave it None if you want AIDB to read token from env variable OPENAI_API_KEY. Otherwise replace None with your own token in str.
  columns_to_input_keys=[('messages', AIDBListType(), 'content')],
  response_keys_to_columns=[('choices', AIDBListType(), 'message', 'content')],
  input_columns_types=[str],
  output_columns_types=[str],
  default_args={"model": "gpt-4-1106-preview",
                ('messages', AIDBListType(), 'role'): "user"},
  prompt_prefix='"',
  prompt_suffix='''"
    The paragraph above is from a law document. 
    Classify whether or not the reasoning of the given paragraph in an opinion is based on the legal framework of textualism. 
    Only answer True or False. Do not provide any reason. Do not add \'.\' in the end.')''')

inference_engines = [
  {
    "service": ocr,
    "input_col": ("pdf.path", "pdf.id"),
    "output_col": ("ocr.text", "ocr.id")
  },
  {
    "service": openai_gpt,
    "input_col": ("ocr.text", "ocr.id"),
    "output_col": ("textualism.label", "textualism.id")
  }
]

blobs_csv_file = "tests/data/law_pdf.csv"
blob_table_name = "pdf"
blobs_keys_columns = ["id"]

"""
dictionary of table names to list of columns
"""

tables = {
  "ocr": [
    {"name": "id", "is_primary_key": True, "refers_to": ("pdf", "id"), "dtype": int},
    {"name": "text", "dtype": str}
  ],
  "textualism": [
    {"name": "id", "is_primary_key": True, "refers_to": ("ocr", "id"), "dtype": int},
    {"name": "label", "dtype": str}
  ]
}
