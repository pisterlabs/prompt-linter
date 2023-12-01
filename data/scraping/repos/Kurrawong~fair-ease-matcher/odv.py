import json
import os
from pathlib import Path

from langchain import PromptTemplate, OpenAI

from src.sparql_queries import find_vocabs_sparql, get_vocabs_from_sparql_endpoint


def read_file(file_path: Path):
    return file_path.read_text()[:10000]


def create_odv_prompt(odv_text):
    template = """\
    The following data is the first 10000 characters from an Ocean Data View file.
    There may be comment lines at the beginning of the file, which start with //.
    I am interested in, for "value" columns:
        1. Vocabularies/concepts used for the columns, these may be specified in columns with a URN, or they may not be specified at all.
        2. Units/concepts for the same columns. These may also be specified with a URN, or not at all, or in the column heading itself or both.
    I am not interested in "Local" URNs. These are of the form "SDN:LOCAL:ABCD". These are only used to map from the comments to the column labels in the data itself.
    I am interested in non "Local" URNs. These are of the form "SDN:P01::ABCDEFG" These refer to external vocabularies.
    I am also interested in Instrument and Observation Platform information if available.
    If a column name is repeated multiple times, it's probably not actually a column - please ignore it.
    Please extract this information based on the columns as JSON in the format below. For each column_name if an 
    attribute ("column_vocabulary_text", "column_vocabulary_urn", "column_unit_text", "column_unit_urn", "instrument", 
    "observation_platform") has information, include that attribute in the response, otherwise do not include it for 
    that column_name. "column_unit_text" is typically included in square brackets for example "[milligram/m3]".
    
    {{
        "columns": [
            {{"column_name" : 
                {{
                    "column_vocabulary_text": "col vocab text",
                    "column_vocabulary_urn": "col vocab urn",
                    "column_unit_text": "col unit text",
                    "column_unit_urn": "col unit urn",
                    "instrument": "instrument text",
                    "observation_platform": "observation platform text"
                }}
            }}
        ]
    }}
    This is the first 10000 characters: {odv_text}
    """
    prompt = PromptTemplate.from_template(template)
    return prompt.format(odv_text=odv_text)


def get_urns_from_odv(odv_json):
    # load json data
    data = json.loads(odv_json)

    # lists to store the vocabulary and unit URNs
    vocab_urns = []
    unit_urns = []

    # go through the columns
    for column in data["columns"]:
        for field in column:
            if "column_vocabulary_urn" in column[field]:
                vocab_urns.append(column[field]["column_vocabulary_urn"])
            if "column_unit_urn" in column[field]:
                unit_urns.append(column[field]["column_unit_urn"])
    if not vocab_urns and not unit_urns:
        raise ValueError("No vocabulary or Unit URNs found")
    return vocab_urns, unit_urns


def main():
    odv_text = read_file(Path("../../data/000545_ODV_77AR2009_00095_H09_V0.txt"))
    prompt = create_odv_prompt(odv_text)
    llm = OpenAI(model_name="gpt-3.5-turbo-0613")
    if os.getenv("TEST_MODE") == "true":
        output = read_file(Path("../../tests/data/odv_response.json"))
    else:
        output = llm(prompt)
    try:
        variable_urns, unit_urns = get_urns_from_odv(output)
        vocab_query = find_vocabs_sparql(variable_urns)
        unit_query = find_vocabs_sparql(unit_urns)
        collections_uris = get_vocabs_from_sparql_endpoint(vocab_query)
        unit_collections_uris = get_vocabs_from_sparql_endpoint(unit_query)
        print(collections_uris, unit_collections_uris)
    except ValueError as e:
        # try next option
        pass


if __name__ == "__main__":
    main()
