import json
import logging
import os

import colorlog
from langchain.docstore.document import Document


def get_logger():
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter("%(log_color)s%(levelname)s:%(name)s:%(message)s")
    )
    logger = colorlog.getLogger("my-logger")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def files_to_documents(folder_name):
    documents = []
    for root, dirs, files in os.walk(folder_name):
        for file in files:
            try:
                with open(os.path.join(root, file), "r", encoding="utf-8") as o:
                    code = o.readlines()
                    d = Document(
                        page_content="\n".join(code),
                        metadata={"source": os.path.join(root, file)},
                    )
                    documents.append(d)
            except UnicodeDecodeError:
                pass
    return documents


def log_model_config(args, logger, log_dir):
    with open(
        os.path.join(log_dir, "config.json"),
        "w",
    ) as f:
        json.dump(vars(args), f)
    logger.info(f"Config stored at {log_dir}/config.json")
