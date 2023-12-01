# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""This module is for loading data """

from langchain.document_loaders import BSHTMLLoader, CSVLoader, PyMuPDFLoader, S3FileLoader, TextLoader


def load_txt_file(file_path: str):
    loader = TextLoader(file_path=file_path)
    return loader.load()


def load_csv_file(file_path: str):
    loader = CSVLoader(file_path=file_path)
    return loader.load()


def load_pdf_file(file_path: str):
    loader = PyMuPDFLoader(file_path=file_path)
    return loader.load()


def load_html_file(file_path: str):
    loader = BSHTMLLoader(file_path=file_path)
    return loader.load()


def load_local_file(file_path: str):
    file_type = file_path.split(".")[-1]
    if file_type == "txt":
        return load_txt_file(file_path)
    elif file_type == "csv":
        return load_csv_file(file_path)
    elif file_type == "pdf":
        return load_pdf_file(file_path)
    elif file_type == "html":
        return load_html_file(file_path)
    else:
        raise NotImplementedError(f"File type {file_type} not implemented.")


def read_txt_file(file_path: str):
    with open(file_path) as f:
        file = f.read()
    return file


def load_from_s3(s3_uri: str):
    if s3_uri.startswith("s3://"):
        s3_uri = s3_uri[5:]
    bucket, key = s3_uri.split("/", 1)
    loader = S3FileLoader(bucket, key)
    return loader.load()

