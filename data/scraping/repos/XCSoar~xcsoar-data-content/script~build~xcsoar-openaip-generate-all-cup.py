#!/bin/env python3
# -*- coding: utf-8 -*-

import requests
import re
import argparse
import os
import json
from iso3166 import countries

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("output", help="Directory to save the file to")
args = parser.parse_args()

base_url = "https://storage.googleapis.com/29f98e10-a489-4c82-ae5e-489dbcd4912f/"
url = base_url
openaip_index = ""
output_directory = os.path.join(args.output, "./content/waypoint/country/")
metajson_directory = "./data/remote/waypoint/country/"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

if not os.path.exists(metajson_directory):
    os.makedirs(metajson_directory)

while True:
    response = requests.get(url)
    xml_data = response.text
    openaip_index += response.text

    # Search for the NextMarker tag in the XML data
    next_marker = None
    match = re.search(r"<NextMarker>(.*?)</NextMarker>", xml_data)
    if match:
        next_marker = match.group(1)
    if next_marker is None:
        break

    url = f"{base_url}?marker={next_marker}"

contents = re.findall(r"<Contents>(.*?)</Contents>", openaip_index)
for content in contents:
    key = re.search(r"<Key>(.*?)</Key>", content)
    metajson_content = False
    if key.group(1).__contains__(".cup"):
        print(key.group(1))
        updatedate = re.search(r"<LastModified>(.*?)</LastModified>", content)
        countrycode = key.group(1)[0:2]
        if key.group(1) == countrycode + "_apt.cup":
            metajson_content = True
            openaipcupfile = open(
                args.output
                + "/content/waypoint/country/"
                + countrycode.upper()
                + "-WPT-"
                + "National"
                + "-OpenAIP.cup",
                "w",
            )
            openaipcupfile.write(requests.get(base_url + key.group(1)).text)
            openaipcupfile.write("\n")
            openaipcupfile.close()
        if key.group(1) == countrycode + "_hgl.cup":
            metajson_content = True
            openaipcupfile = open(
                args.output
                + "/content/waypoint/country/"
                + countrycode.upper()
                + "-WPT-"
                + "National"
                + "-OpenAIP.cup",
                "a",
            )
            openaipcupfile.write(requests.get(base_url + key.group(1)).text)
            openaipcupfile.write("\n")
            openaipcupfile.close()
        if key.group(1) == countrycode + "_hot.cup":
            metajson_content = True
            openaipcupfile = open(
                args.output
                + "/content/waypoint/country/"
                + countrycode.upper()
                + "-WPT-"
                + "National"
                + "-OpenAIP.cup",
                "a",
            )
            openaipcupfile.write(requests.get(base_url + key.group(1)).text)
            openaipcupfile.write("\n")
            openaipcupfile.close()
        if key.group(1) == countrycode + "_nav.cup":
            metajson_content = True
            openaipcupfile = open(
                args.output
                + "/content/waypoint/country/"
                + countrycode.upper()
                + "-WPT-"
                + "National"
                + "-OpenAIP.cup",
                "a",
            )
            openaipcupfile.write(requests.get(base_url + key.group(1)).text)
            openaipcupfile.write("\n")
            openaipcupfile.close()
        if key.group(1) == countrycode + "_rpp.cup":
            metajson_content = True
            openaipcupfile = open(
                args.output
                + "/content/waypoint/country/"
                + countrycode.upper()
                + "-WPT-"
                + "National"
                + "-OpenAIP.cup",
                "a",
            )
            openaipcupfile.write(requests.get(base_url + key.group(1)).text)
            openaipcupfile.write("\n")
            openaipcupfile.close()

        if metajson_content:
            metajson = {}
            metajson["uri"] = (
                "https://download.xcsoar.org/content/waypoint/country/"
                + countrycode.upper()
                + "-WPT-National-OpenAIP.cup"
            )
            metajson["description"] = (
                str(countries.get(countrycode).apolitical_name)
                + " aviation data from OpenAIP"
            )
            metajsonfile = open(
                "data/remote/waypoint/country/"
                + countrycode.upper()
                + "-WPT-National-OpenAIP.cup.json",
                "w",
                encoding="utf-8",
            )
            json.dump(metajson, metajsonfile, ensure_ascii=False, indent=2)
            metajsonfile.write("\n")
            metajsonfile.close()
