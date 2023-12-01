# How do we retrieve information from json files?
# jq lang: https://jqlang.github.io/jq/
# jq playground: https://jqplay.org/s/U0LctMZh6gj

import pprint
from langchain.document_loaders import JSONLoader

loader = JSONLoader(
    file_path='data/movies_embedding_short.json',
    jq_schema='.{"title", "release_date"}') # fixme

data = loader.load()

pprint(data)