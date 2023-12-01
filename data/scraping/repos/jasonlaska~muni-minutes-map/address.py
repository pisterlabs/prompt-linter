import os
import openai
import json
from tqdm import tqdm
from dataclasses import dataclass, asdict

from . import get_openai_key
from .base import DocumentProcessor
from .pdf2text import TextPage
from ..artifact import Artifact


PROCESSOR_NAME = "address"

SYSTEM_NUDGE = (
    "You are a helpful document parsing tool that only responds in json objects."
)
PROMPT = """Find property addresses in the document for which there is a memorialization.
Return the addresses as a json object like:

    {{
        "addresses": [<insert address 0>, <insert address 1>, ...],
    }}

Rules:
1. Process each property address independently
2. Think step by step for each property
3. If you cannot find addresses return None (and no other text or information)

Document: {text}"""
MAX_TOKENS = 1500
TEMPERATURE = 0

GPT_FALSE_ADDRESS_FILTER = [
    "none",
    "",
    "no addresses found",
    "no address found",
    "subject property",
]


@dataclass
class AddressDetection:
    street: str
    aliases: list[str]
    pages: list[TextPage]


class Address(DocumentProcessor):
    def __init__(self, source):
        self._source = source
        self._errors = []
        self._addresses = None
        self._artifact = Artifact(source, PROCESSOR_NAME)

        openai.api_key = get_openai_key()

    @property
    def errors(self):
        return self._errors

    @property
    def result(self):
        return self._addresses

    @property
    def artifact_exists(self):
        return self._artifact.exists

    def _merge(self, addresses):
        merged = dict()
        for address in addresses:
            street = address["address"].split(",")[0].lower()

            if street not in merged:
                merged[street] = AddressDetection(
                    street, [address["address"]], [address["page"]]
                )

            else:
                merged[street].aliases.append(address["address"])
                merged[street].pages.append(address["page"])

        return merged

    def _filter(self, addresses):
        return [
            a for a in addresses if a["address"].lower() not in GPT_FALSE_ADDRESS_FILTER
        ]

    def _extract_from_text(self, pages):
        addresses = []
        for page in tqdm(pages, desc=f"Addresses from {len(pages)} pages..."):
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_NUDGE,
                    },
                    {"role": "user", "content": PROMPT.format(text=page.text)},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            result = response["choices"][0]["message"]["content"]

            try:
                data = json.loads(result)
                addresses += [
                    {"address": a, "page": page.page} for a in data["addresses"]
                ]
            except:
                self._errors.append("INVALID_JSON")

        return addresses

    def extract(self, pages):
        # Input:    [TextPage, ...]
        # Returns:  {street: AddressDetection, ...}
        self._addresses = self._merge(self._filter(self._extract_from_text(pages)))
        return self._addresses

    def save(self, overwrite=False):
        return self._artifact.write(
            {k: asdict(v) for k, v in self._addresses.items()}, overwrite=overwrite
        )

    def load(self):
        self._addresses = {
            k: AddressDetection(**v) for k, v in self._artifact.read().items()
        }
        return self._addresses
