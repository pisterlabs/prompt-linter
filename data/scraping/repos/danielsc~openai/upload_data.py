
import os, argparse, uuid, yaml
import openai
from azureml.core import Run

parser = argparse.ArgumentParser()
parser.add_argument("--file_to_upload", default="./data/2validation/yelp_validation.jsonl")
parser.add_argument("--upload_metadata", default="./data/3validation_uploaded")
parser.add_argument("--aoai_resource_name", default="aoai")
parser.add_argument("--api_version", default="2022-06-01-preview")

args = parser.parse_args()

def save_yaml(content, filename):
   with open(filename, encoding='utf-8', mode='w') as fh:
      yaml.dump(content, fh)

def load_api_key(keyname = "openai-key"):
  from azureml.core.run import _OfflineRun
  from azureml.core import Run
  run = Run.get_context()
  if run.__class__ == _OfflineRun:
    print(f"loading key {keyname} from environment variables")
    secret_value = os.getenv("OPENAI_API_KEY")
  else:
    print(f"loading key {keyname} from keyvault")
    secret_value = run.get_secret(name=keyname)

  if not secret_value is None:
    return secret_value
  else:
    raise Exception("No OPENAI_API_KEY found in environment variables or keyvault")

def upload_file(file):
  # extract file name from path in on os-agnostic way
  file_name = os.path.basename(file)

  upload_result = openai.File.create(
    file=open(file, "rb"),
    user_provided_filename=file_name + ".jsonl",
    purpose='fine-tune'
  )
  print(upload_result)
  return upload_result

openai.api_key = load_api_key()
openai.api_base = f"https://{args.aoai_resource_name}.openai.azure.com/" # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
openai.api_type = 'azure'
openai.api_version = args.api_version # this may change in the future

## upload the files
print("Uploading data")
metadata = upload_file(args.file_to_upload)
print("success")
print(metadata)

print("writing metadata to output")
save_yaml(metadata.to_dict(), args.upload_metadata + "/MLArtifact.yaml")

