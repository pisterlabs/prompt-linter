from .application import app
from flask import render_template, Response, send_from_directory, make_response, redirect
import flask
from http import HTTPStatus
from flask import request
import os
import openai
from utils.chains import *
from utils.openai_utils import *
from utils.file_utils import *
from utils.legacy_file_utils import *
from utils.agent import *

server = "https://cbfd-2405-201-35-f061-dd7a-162b-c6d0-ff0c.ngrok-free.app"
generator = ContextBasedGenerator()
def get_pdf_from_client(file):
    if(file.filename.endswith(".pdf")):
        file_path = os.path.join("storage/pdfs",file.filename)
        file.save(file_path)
    else:
        raise Exception("File must be a pdf")
    return file_path

def get_gpt_response(prompt, pdf_paths):
    generator.generate_db_from_pdf(pdf_paths)
    gpt_response = generator.generate_chain_response(prompt)
    print("obtained response from gpt ")
    response_json = gpt_response.json()
    print(response_json)
    return response_json["choices"][0]["message"]["content"]

# def get_gpt_response(prompt, pdf_paths):
#     generator_new = ContextBasedGeneratorAgent(pdf_paths)
#     # generator.generate_db_from_pdf(pdf_paths)
#     # print("Initialized generator")
#     # gpt_response = generator.generate_chain_response(prompt)
#     gpt_response_new = generator_new.generate_chain_response(prompt)
#     print("obtained response from gpt")
#     # call extra processors if needed
#     # return gpt_response[0]["text"]
#     return gpt_response_new
