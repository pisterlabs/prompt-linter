import json
import os
import csv
from dataclasses import dataclass
from typing import List
import re

from langchain.document_loaders import CSVLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredExcelLoader


@dataclass(frozen=True)
class ValidateFileResult:
    is_valid: bool
    error_message: str = None


@dataclass(frozen=True)
class ValidateCopilotResult:
    is_valid: bool
    error_results: List[ValidateFileResult]


def execute(relative_directory_path: str) -> ValidateCopilotResult:
    file_paths = [
        os.path.join(dp, f)
        for dp, dn, filenames in os.walk(relative_directory_path)
        for f in filenames
    ]
    error_results = _validate_required_files(relative_directory_path)
    error_results += _validate_prompt_template(relative_directory_path)
    for file_path in file_paths:
        if "/.git/" not in file_path:
            result = _validate_file(file_path)
            if not result.is_valid:
                error_results.append(result)

    if error_results:
        return ValidateCopilotResult(is_valid=False, error_results=error_results)
    return ValidateCopilotResult(is_valid=True, error_results=[])


def _validate_required_files(relative_directory_path: str) -> List[ValidateFileResult]:
    result = []
    files_to_check = [
        "prompts/prompt_template.txt",
        "prompts/prompt_configuration.json",
    ]
    for file in files_to_check:
        file_path = os.path.join(relative_directory_path, file)
        if not os.path.exists(file_path):
            result.append(
                ValidateFileResult(is_valid=False, error_message=f"Missing {file} file")
            )
    return result


def _validate_prompt_template(relative_directory_path: str) -> List[ValidateFileResult]:
    required_params = ["{context}", "{history}", "{question}"]
    file_path = os.path.join(relative_directory_path, "prompts/prompt_template.txt")
    if os.path.exists(file_path):
        content = _read_file(file_path)
        for param in required_params:
            count = _find_word_count_in_string(param, content)
            if count == 0:
                return [
                    ValidateFileResult(
                        is_valid=False,
                        error_message="Prompt template (prompts/prompt_template.txt) is missing {context}"
                        ", {history} or {question}",
                    )
                ]
            elif count > 1:
                return [
                    ValidateFileResult(
                        is_valid=False,
                        error_message="Prompt template (prompts/prompt_template.txt) is having duplicate of {context}, "
                        "{history} or {question}",
                    )
                ]
    return []


def _find_word_count_in_string(word: str, content: str) -> int:
    count = 0
    for _ in re.finditer(word, content):
        count += 1
    return count


def _validate_file(file_path: str) -> ValidateFileResult:
    ext = os.path.splitext(file_path)[1]
    if ext == ".json":
        try:
            loader = TextLoader(file_path)
            loader.load()
            content = _read_file(file_path)
            json.loads(content)
        except:
            return ValidateFileResult(
                is_valid=False, error_message=f"Cannot load json file: {file_path}"
            )
    elif ext == ".csv":
        try:
            with open(file_path, newline="") as csvfile:
                dialect = csv.Sniffer().sniff(csvfile.read(1024))

            loader = CSVLoader(
                file_path=file_path,
                csv_args={
                    "delimiter": dialect.delimiter,
                },
            )
            loader.load()
        except:
            return ValidateFileResult(
                is_valid=False, error_message=f"Cannot csv json file: {file_path}"
            )
    elif ext == ".pdf":
        try:
            loader = PyPDFLoader(file_path)
            loader.load()
        except:
            return ValidateFileResult(
                is_valid=False, error_message=f"Cannot load pdf file: {file_path}"
            )
    elif file_path.endswith(".xls") or file_path.endswith(".xlsx"):
        try:
            loader = UnstructuredExcelLoader(file_path)
            loader.load()
        except:
            return ValidateFileResult(
                is_valid=False, error_message=f"Cannot load excel file: {file_path}"
            )
    return ValidateFileResult(is_valid=True)


def _read_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()
