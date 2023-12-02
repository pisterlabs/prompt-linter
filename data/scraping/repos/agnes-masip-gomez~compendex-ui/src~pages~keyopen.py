import openai
import base64
import os
encode_string= #"Put the key here"
decode_string = base64.b64decode(encode_string).decode()
os.environ["OPENAI_API_KEY"] = decode_string 
openai.api_key = os.getenv("OPENAI_API_KEY")
import traceback

print(decode_string)