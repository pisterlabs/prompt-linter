from dotenv import load_dotenv
import openai
import json
from os import environ
from pprint import pprint

import requests

load_dotenv()

openai.api_key = environ["OPENAI_API_KEY"]
name_token_dev = environ["NAME_TOKEN_DEV"]
name_user_dev = environ["NAME_USER_DEV"]
name_token_prod = environ["NAME_TOKEN"]
name_user = environ["NAME_USER"]

name_url_dev = "https://api.dev.name.com/v4/"
name_url_prod = "https://api.name.com/v4/"


def get_company_name_for(description):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=generate_prompt(description),
        temperature=0,
        n=1,
    )
    return response["choices"][0].get("text").split(",")


def generate_prompt(description):
    return """
        Suggest three names for a company based on these descriptors.
        Name should be one word long.
        Description: {}
        Only respond with the three names, with a comma and space between each.
        For example:
            Name 1, Name 2, Name 3
    """.format(
        description.capitalize()
    )


def query_name_for_domains(phrase):
    print(phrase)
    params = f'{{"keyword":"{phrase}"}}'
    headers = {"Content-Type": "application/json"}
    url = f"{name_url_prod}domains:search"

    print(url)

    r = requests.post(
        url, data=params, auth=(name_user, name_token_prod), headers=headers
    )
    print(r.content)
    return json.loads(r.content)


def is_domain_free(domains):
    allowed_tlds = ["com", "net", "io", "co", "ai", "info", "xyz", "org"]
    free_domains = [
        {
            "domain": d.get("domainName"),
            "available": d.get("purchasable"),
            "price": f"{d.get('purchasePrice')} USD",
        }
        for d in domains.get("results", [])
        if d.get("purchasable") is True and d.get("tld") in allowed_tlds
    ]
    if free_domains:
        return free_domains
    else:
        return []


def get_all_domains(phrase):
    companies = get_company_name_for(phrase)
    domains = [query_name_for_domains(c) for c in companies]
    free_domains = [is_domain_free(d) for d in domains]
    return free_domains


if __name__ == "__main__":
    pprint(get_all_domains("cars for clowns"))
