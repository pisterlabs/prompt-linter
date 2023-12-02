import os
import random
from typing import Optional

import openai
from dotenv import load_dotenv


def set_keys_from_env():
    # take environment variables from .env so you don't have
    # to source .env in your shell
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = api_key


def get_org_ids() -> Optional[list[str]]:
    ids = os.getenv("OPENAI_ORG_IDS", None)
    if ids is None:
        return
    ids = ids.split(",")
    return ids


def set_opeani_org_from_env_rand():
    ids = os.getenv("OPENAI_ORG_IDS", None)
    if ids is None:
        return
    ids = ids.split(",")
    if len(ids) > 1:
        id_ = random.choice(ids)
    else:
        id_ = ids[0]
    openai.organization = id_
