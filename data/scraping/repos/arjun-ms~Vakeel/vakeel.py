from langchain.document_loaders import JSONLoader

from langchain.docstore.document import Document

import json
from pathlib import Path
from pprint import pprint


file_path='ipc_id.json'

# data = json.loads(Path(file_path).read_text(encoding="utf-8"))
# pprint(data)

loader = JSONLoader(
    file_path=file_path,
    jq_schema='.messages[].content')

data = loader.load()
pprint(data)