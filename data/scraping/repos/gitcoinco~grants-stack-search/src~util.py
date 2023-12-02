import re
from eth_utils.address import to_checksum_address
from urllib.parse import urljoin
from langchain.schema import Document
from typing import List, Self
from dataclasses import dataclass
from pythonjsonlogger import jsonlogger
from pythonjsonlogger.jsonlogger import JsonFormatter


class InvalidInputDocumentException(Exception):
    pass


# Wrap a Document to carry validity information together with the type
class InputDocument:
    def __init__(self, document: Document):
        if (
            isinstance(document.metadata.get("application_ref"), str)
            and isinstance(document.metadata.get("name"), str)
            and isinstance(document.metadata.get("website_url"), str)
            and isinstance(document.metadata.get("round_id"), str)
            and isinstance(document.metadata.get("chain_id"), int)
            and isinstance(document.metadata.get("round_application_id"), str)
            and isinstance(document.metadata.get("payout_wallet_address"), str)
            # banner_image_cid and logo_image_cid are optional
            and document.page_content is not None
        ):
            self.document = document
        else:
            raise InvalidInputDocumentException(
                document.metadata.get("application_ref")
            )


@dataclass
class ApplicationFileLocator:
    chain_id: int
    round_id: str

    @classmethod
    def from_string(cls, application_file_locator_s: str) -> Self:
        match = re.match(r"^(\d+):(0x[0-9a-fA-F]+)$", application_file_locator_s)
        if not match:
            raise Exception(
                f"Invalid application locator: {application_file_locator_s}"
            )

        return cls(
            chain_id=int(match.group(1)), round_id=to_checksum_address(match.group(2))
        )


def parse_applicaton_file_locators(
    application_file_locators_s: str,
) -> List[ApplicationFileLocator]:
    return list(
        map(ApplicationFileLocator.from_string, application_file_locators_s.split(","))
    )


def get_rounds_file_url_from_chain_id(chain_id: int, indexer_base_url: str) -> str:
    return urljoin(
        indexer_base_url,
        f"/data/{chain_id}/rounds.json",
    )


def get_applications_file_url_from_application_file_locator(
    application_file_locator: ApplicationFileLocator, indexer_base_url: str
) -> str:
    return urljoin(
        indexer_base_url,
        f"/data/{application_file_locator.chain_id}/rounds/{application_file_locator.round_id}/applications.json",
    )


def get_json_log_formatter(hostname: str, deployment_environment: str) -> JsonFormatter:
    return jsonlogger.JsonFormatter(
        # TODO: add `created` as Unix time in milliseconds (the available
        # `created` field is in milliseconds). Could be done by subclassing
        # JsonFormatter, see https://stackoverflow.com/a/52933068
        "[%(levelname)8s] %(message)s %(filename)s:%(lineno)d %(asctime)",
        rename_fields={"levelname": "level"},
        static_fields={
            "hostname": hostname,
            "service": f"search-{deployment_environment}",
        },
    )
