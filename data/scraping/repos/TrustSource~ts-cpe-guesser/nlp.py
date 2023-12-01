"""
# Parser
This module enables parsing natural language descriptions of vulnerabilities for vendor,
product and version info. Currently, only one backend model is available: openai.

## Sample usage:

```
    import ts_vulndb.parser.generate_cpe_match as generate_cpe_match

    vulnerability_description = open("vuln.txt","r").read()
    vulnerability_info = generate_cpe_match(vulnerability_description)

    vendor = vulnerability_info["vendor"]
    product = vulnerability_info["product"]
    original_text = vulnerability_info["description"]

    do_something(vendor,product,original_text) ..
```

"""
from typing import Union

import openai
import re
import time
import utils

def generate_cpe_match(description: Union[str, list], method: str = "openai",
                       verbose: bool = True) -> Union[dict, list]:
    """
    Extracts vendor, product and version information from a text describing a vulnerability
    that affects a specific vendor/product/version configuration.

    Args:
        description (str) / (list):
            A vulnerability description ideally containing one word that denotes a vendor,
            such as "Jetbrains", a product such as "PyCharm" and a version range.
        method (str):
            Which backend to use in order to generate cpe data from the text. Can be "openai" or "nltk".
                --"openai"-- sends a request to a trained model hosted on OpenAI's server. Each prompt
            costs approximately 0.5c. Maximum requests per minute: 60.
                --"nltk"-- can be run locally without any restrictions or additional costs.
        verbose (bool):
            print model details to console, if true.
    Returns:
        generated_data (Union[list,dict]):
            dictionaries with guaranteed keys:
            ["vendor", "product", "version", "description", "nlp_method"],
            optional keys:
            ["versionStartExcluding", "versionStartIncluding", "versionEndExcluding", "versionEndIncluding",
    """
    if type(description) == list:
        generated_batch = generate_cpe_matches(description, method)
        return generated_batch
    if method == "openai":
        generated_data = openai_parse(description, verbose)
        nlp_method = {"framework": "openai", "model": "curie:ft-eacg:nvd21-665ner-2022-07-19-15-01-06"}
    elif method == "nltk":
        generated_data = nltk_parse(description)
        nlp_method = {"framework": "nltk"}
    else:
        raise ValueError("Invalid 'method'. Must be one of ['openai','nltk']")

    if generated_data:
        generated_data["cpe_uri"] = make_cpe_uri(generated_data)
    else:
        generated_data["cpe_uri"] = "cpe:2.3:-:-:-:-:-:-:-:-:-:-:-"

    generated_data["description"] = description  # To link
    # the original text with the generated entities.
    generated_data["nlp_method"] = nlp_method
    return generated_data


def generate_cpe_matches(descriptions: [str], method: str = "openai",
                         verbose: bool = False) -> [str]:
    """
    Wrapper for generate_cpe_match, in case multiple descriptions must be processed.

    Args:
        descriptions ([str]):
            List of vulnerability descriptions ideally featuring vendor, product and version info
        method (str):
            Method used to generate cpe info from the text ["openai","nltk"]
        verbose (bool):
            If true, details about the generation process are printed to the console.

    Returns:
        generated_batch - Keys of each item
            ["vendor","product","version", "versionStartExcluding",
            "versionStartIncluding", "versionEndExcluding", "versionEndIncluding",
            "description", "nlp_method"]
    """
    try:
        assert method in ["openai", "nltk"]
    except Exception:
        raise ValueError("Invalid value for 'method' parameter. Must be one of ['openai','nltk']")

    generated_batch = []
    print(f"Extracting information from descriptions using method '{method}'..")
    for i, description in enumerate(descriptions):
        if not verbose:
            print(f"{i + 1}/{len(descriptions)}", end="\r")  # progress report
        else:
            print(f"{i + 1}/{len(descriptions)} Completions")
        if method == "openai" and len(descriptions) > 10:  # bypassing openAI's rate limit
            time.sleep(1.2)
        result = generate_cpe_match(description, method, verbose=verbose)
        generated_batch.append(result)
    print("Done   ")

    return generated_batch


def openai_parse(description: str, verbose: bool = True) -> dict:
    """
    Creates a prompt, sends it to a pretrained model on openaiAPI and formats the
    api response into a dictionary.

    Args:
        description (str): Natural Language Vulnerability Description
        verbose (bool): Determines whether completions are printed to the console.
    Returns:
        formatted_entities (dict):
            Dict with vendor, product and version info.
            Keys: ["vendor","product","version","{start|end}_{excluding|including}"]

            Empty dict ({}), when API Call or formatting failed
    """
    openai.api_key = utils.get_credentials("dev/cpe-guesser/openai-apikey")  # Hard-coded for now.
    model = "curie:ft-eacg:nvd21-665ner-2022-07-19-15-01-06"  # Hard-coded for now.
    indicator_string = "\n->"
    prompt = description + indicator_string
    try:
        if verbose:
            print(f"Making API Call to Openai Model\n{model}")
        completion = openai.Completion.create(
            model=model,
            prompt=prompt,
            temperature=0.1,
            max_tokens=32,
            stop=[",-,", "\n-\n"]
        )
    except Exception as E:
        print(E)
        print("Request to openAI API failed.")
        return {}

    try:
        if verbose:
            print(f"Received OpenAI API completion:\n{completion}\n")
            print("Formatting completion..")

        # Completion looks like "vendor,product,version,<version,<=version,>version,>=version"
        text = completion.choices[0].text
        text = re.sub(" ", "", text)
        # Entities are comma-separated in the completion string.
        generated_entities = text.split(",")
    except Exception as E:
        print(E)
        print("Failed to extract entities from response string. ")
        return {}

    # This is the return Value
    formatted_entities = {}

    # The order matters. Background knowledge about the training data for the
    # model allows to infer the order in which the entities are listed in the
    # model output.
    entity_names = ["vendor", "product", "version",
                    "versionEndExcluding", "versionEndIncluding",
                    "versionStartExcluding", "versionStartIncluding"]
    for i in range(len(entity_names)):
        name = entity_names[i]
        try:
            generated = generated_entities[i]
            formatted = re.sub("[*<>,:=-]", "", generated)  # Strip non-word characters
            if formatted.lower() in ["none", ""]:
                continue  # skipping 'nones'
            formatted_entities[name] = formatted

        except IndexError:
            if verbose:
                print(f"Fewer entities than expected listed in openAI response.")
                print(f"No problem. Stopped before extracting '{name}' ")
            break

    # Handling Missing Version values
    version_limits = ["versionStartExcluding",
                      "versionStartIncluding",
                      "versionEndExcluding",
                      "versionEndIncluding"]
    # Enforce the existence of "vendor", "product" and "version" keys
    if "vendor" not in formatted_entities.keys():
        formatted_entities["vendor"] = "-"
    if "product" not in formatted_entities.keys():
        formatted_entities["product"] = "-"
    # If there is no "version" key in formatted
    if "version" not in formatted_entities.keys():
        # but there is a version limit indicator, set version to "*"
        if len(set(version_limits) & set(formatted_entities.keys())) > 0:
            formatted_entities["version"] = "*"
        else:
            formatted_entities["version"] = "-"

    return formatted_entities


def nltk_parse(description) -> dict:
    return {"Method not implemented": ""}


def make_cpe_uri(config: dict) -> str:
    """
    Creates a CPE Uri from a 'config' as received by nlp.generate_cpe_match
    Args:
        config: dict with keys ["vendor","product","version"]
    Returns:
        cpe_uri: well-formed CPE 2.3 string featuring vendor, product and version
    """
    vendor = config["vendor"]
    product = config["product"]
    version = config["version"]
    cpe_uri = f"cpe:2.3:*:{vendor}:{product}:{version}:*:*:*:*:*:*:*"
    return cpe_uri
