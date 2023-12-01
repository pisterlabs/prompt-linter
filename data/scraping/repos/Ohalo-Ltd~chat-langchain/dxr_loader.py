# following tutorial from https://github.com/Coding-Crashkurse/LangChain-Custom-Loaders
import os
import requests
from langchain.docstore.document import Document
from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders.base import BaseLoader
from typing import Any, Dict, List, Optional

# list of classes that should be redacted by default
DEFAULT_PII_REDACTION_PROFILE = [53]


@staticmethod
def redactor(s, start_index, end_index):
    redaction_length = end_index - start_index
    redaction_val = "X" * redaction_length

    # insert the new string between "slices" of the original
    return s[:start_index] + redaction_val + s[end_index:]


@staticmethod
def redact_text(
    dxr_api_url: str, text: str, redacted_annotation: List[int], bearer_token: str
) -> Optional[str]:
    auth_header_value = f"Bearer {bearer_token}"
    headers = {
        "Accept": "application/json, text/plain, */*",
        "Authorization": auth_header_value,
        "Content-Type": "application/json",
    }

    dxr_api_nlp_url = f"{dxr_api_url}/nlp"

    json_data = {"dataClassIds": redacted_annotation, "inputText": text}
    
    response = requests.post(dxr_api_nlp_url, headers=headers, json=json_data)
    response_body = response.json()
    
    input_text = ""

    if not response_body.get("input_text"):
        return
    else:
        input_text = response_body["input_text"]

    for a in response_body["annotations"]:
        input_text = redactor(input_text, a[1], a[2])

    return input_text


@staticmethod
def get_documents(dxr_api_url: str, dxr_label: int, bearer_token: str) -> Optional[str]:
    dxr_api_search_url = f"{dxr_api_url}/indexed-files/search"

    auth_header_value = f"Bearer {bearer_token}"

    headers = {
        "Accept": "application/json, text/plain, */*",
        "Authorization": auth_header_value,
        "Content-Type": "application/json",
    }

    json_data = {
        "mode": "DXR_JSON_QUERY",
        "datasourceIds": [],
        "pageNumber": 0,
        "pageSize": 20,
        "filter": {
            "query_items": [
                {
                    "parameter": "dxr#tags",
                    "value": dxr_label,
                    "type": "number",
                    "match_strategy": "exact",
                    "operator": "AND",
                    "group_id": 0,
                    "group_order": 0,
                },
            ],
        },
        "sort": [
            {
                "property": "_score",
                "order": "DESCENDING",
            },
        ],
        "minScore": 0,
    }

    response = requests.post(dxr_api_search_url, headers=headers, json=json_data)

    try:
        return response.json()["hits"]["hits"]
    except Exception as e:
        print(f"Error while parsing response to JSON: {e}")
        raise e


class DataXRayLoader(BaseLoader):
    def __init__(
        self,
        dxr_url,
        dxr_label,
        data_protection_shield_enabled=False,
        data_protection_shield_profile=DEFAULT_PII_REDACTION_PROFILE,
        **kwargs,
    ):
        # for now I'm just going to wrap webbased loader
        self.dxr_url = dxr_url
        self.dxr_label = dxr_label
        self.data_protection_shield_enabled = data_protection_shield_enabled
        self.data_protection_shield_profile = data_protection_shield_profile
        self.apiKey = os.environ["DXR_API_KEY"]

    def load(self):
        docs = []
        raw_docs = get_documents(self.dxr_url, self.dxr_label, self.apiKey)

        for d in raw_docs:
            source = d["_source"]

            if not source.get("dxr#raw_text"):
                continue

            content = source["dxr#raw_text"]
            datasource_id = source["datasource_id"]
            object_id = source["object_id"]

            # Redact each document on the fly
            if self.data_protection_shield_enabled:
                content = redact_text(
                    self.dxr_url,
                    content,
                    self.data_protection_shield_profile,
                    self.apiKey,
                )

            doc = Document(
                page_content=content,
                metadata={"source": f"{datasource_id}___{object_id}"},
            )
            docs.append(doc)

        return docs
