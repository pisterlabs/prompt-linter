import openai
import pprint
import configparser

config = configparser.ConfigParser()
config.read(".env")
openai.api_key = config["keys"]["OPENAI_API_KEY"]
openai.organization = config["keys"]["OPENAI_ORG_KEY"]


def list_all_models():
    model_list = openai.Model.list()["data"]
    model_ids = [x["id"] for x in model_list]
    model_ids.sort()
    pprint.pprint(model_ids)
    if "gpt-4-32k" in model_ids:
        print("##################################################")
        print("##################################################")
        print("##################################################")
        print()
        print("gpt-4-32k IS PUBLISHED")
        print()
        print("##################################################")
        print("##################################################")
        print("##################################################")
    else:
        print()
        print()
        print("gpt-4-32k is not out yet :(")


list_all_models()
