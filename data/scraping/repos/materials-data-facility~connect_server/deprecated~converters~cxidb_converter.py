import json
import sys
import os

from tqdm import tqdm

from mdf_refinery.validator import Validator
from mdf_forge.toolbox import find_files

# VERSION 0.3.0

# This is the converter for CXIDB.
# Arguments:
#   input_path (string): The file or directory where the data resides.
#       NOTE: Do not hard-code the path to the data in the converter. The converter should be portable.
#   metadata (string or dict): The path to the JSON dataset metadata file, a dict or json.dumps string containing the dataset metadata, or None to specify the metadata here. Default None.
#   verbose (bool): Should the script print status messages to standard output? Default False.
#       NOTE: The converter should have NO output if verbose is False, unless there is an error.
def convert(input_path, metadata=None, verbose=False):
    if verbose:
        print("Begin converting")

    # Collect the metadata
    if not metadata:
        dataset_metadata = {
        "mdf": {
            "title": "The Coherent X-ray Imaging Data Bank",
            "acl": ["public"],
            "source_name": "cxidb",
            "citation": ["Maia, F. R. N. C. The Coherent X-ray Imaging Data Bank. Nat. Methods 9, 854–855 (2012)."],
            "data_contact": {

                "given_name": "Filipe",
                "family_name": "Maia",

                "email": "cxidb@cxidb.org",
                "institution": "Lawrence Berkeley National Laboratory",

                # IDs
                },

            "author": {

                "given_name": "Filipe",
                "family_name": "Maia",

                "institution": "Lawrence Berkeley National Laboratory",

                # IDs
                },

#            "license": ,

            "collection": "CXIDB",
            "tags": ["x-ray", "coherent"],

            "description": "A new database which offers scientists from all over the world a unique opportunity to access data from Coherent X-ray Imaging (CXI) experiments.",
            "year": 2012,

            "links": {

                "landing_page": "http://www.cxidb.org/",

#                "publication": ,
#                "dataset_doi": ,

#                "related_id": ,

                # data links: {

                    #"globus_endpoint": ,
                    #"http_host": ,

                    #"path": ,
                    #}
                },

#            "mrr": ,

            "data_contributor": {
                "given_name": "Jonathon",
                "family_name": "Gaff",
                "email": "jgaff@uchicago.edu",
                "institution": "The University of Chicago",
                "github": "jgaff"
                }
            }
        }
    elif type(metadata) is str:
        try:
            dataset_metadata = json.loads(metadata)
        except Exception:
            try:
                with open(metadata, 'r') as metadata_file:
                    dataset_metadata = json.load(metadata_file)
            except Exception as e:
                sys.exit("Error: Unable to read metadata: " + repr(e))
    elif type(metadata) is dict:
        dataset_metadata = metadata
    else:
        sys.exit("Error: Invalid metadata parameter")


    dataset_validator = Validator(dataset_metadata)


    # Get the data
    for dir_data in tqdm(find_files(input_path, file_pattern="json", verbose=verbose), desc="Processing metadata", disable= not verbose):
        with open(os.path.join(dir_data["path"], dir_data["filename"])) as file_data:
            cxidb_data = json.load(file_data)
        record_metadata = {
        "mdf": {
            "title": cxidb_data["citation_title"],
            "acl": ["public"],

#            "tags": ,
#            "description": ,
            
#            "composition": ,
            "raw": json.dumps(cxidb_data),

            "links": {
                "landing_page": cxidb_data["url"],

                "publication": [cxidb_data.get("citation_DOI", None), cxidb_data.get("entry_DOI", None)],
#                "dataset_doi": ,

#                "related_id": ,

                # data links: {
 
                    #"globus_endpoint": ,
                    #"http_host": ,

                    #"path": ,
                    #},
                },

#            "citation": ,
#            "data_contact": {

#                "given_name": ,
#                "family_name": ,

#                "email": ,
#                "institution":,

                # IDs
#                },

#            "author": ,

#            "license": ,
#            "collection": ,
#            "data_format": ,
#            "data_type": ,
#            "year": ,

#            "mrr":

#            "processing": ,
#            "structure":,
            }
        }

        # Pass each individual record to the Validator
        result = dataset_validator.write_record(record_metadata)

        # Check if the Validator accepted the record, and print a message if it didn't
        # If the Validator returns "success" == True, the record was written successfully
        if result["success"] is not True:
            print("Error:", result["message"])


    if verbose:
        print("Finished converting")
