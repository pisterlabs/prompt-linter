import openai
import argparse
import os
import json

from src.summarizer_class import File_Summarizer
from src.file_io import get_target_path
from src.pathtools_class import Path_Tools

'''
This script will summarize an entire directory specified with --location, or all the contents of target_folder.
'''

def main():
    global args 
    args = arg_parse() #file_location and focus_question
    file_sum = File_Summarizer()

    # setup openai key
    try :
        api_key = os.environ["OPENAI_API_KEY"]
    except:
        api_key = file_sum.get_key()
    openai.api_key = api_key

    target_path = get_target_path(args.target_path) # target directory

    Path_Tools.print_testing_summary(target_path)

    

def print_summaries(summary_string, summary_list, open_api):
    # prints out full response to terminal

    #! impliment newlines to make response more readable!
    print(f"GPTs full response to {open_api.question}:")
    print(summary_string)
    print(f"GPT has {len(summary_list)} responses. Each one is broken down below:")

    for i in range(len(summary_list)):
        
        print(f"\nresponse {i}:")
        print(summary_list[i])

def arg_parse():
    # arguements for python script
    parser = argparse.ArgumentParser(description="GTP3 File Summarizer")
    parser.add_argument("--target_path", type=str, dest="target_path", help="Path to target directory to be analzyed", default="")
    parser.add_argument("--question_type", type=int, dest="question_type", help="Which question to ask. See README and questions.txt", default=0)
    # parser.add_argument("--model_name", type=str, dest="model_name", help="engine used to make requests from", default="gpt-3.5-turbo-0301")
    parser.add_argument("--response_size", type=int, dest="response_size", help="maximum number of works in final response. 50 to max tokensize", default=0)
    parser.add_argument("--temp", type=int, dest="temp", help="randomness of models response. From 0 to 1", default=0.5)
    parser.add_argument("--overlap", type=int, dest="overlap", help="overlap is the amount of text repeated in a summarization of content, to make sure context isn't lost.", default=150)
    
    args = parser.parse_args()
    prepare_json(args)

    return args

def prepare_json(args):
    # changes a json file using commandline arguements that have been parsed 

    json_filename = "api_params.json"

    cur_dir = os.getcwd()
    json_path = cur_dir + "/src/" + json_filename

    with open(json_path, 'r') as f:
        params = json.load(f)

    params["temperature"] = args.temp
    params["file_location"] = args.target_path

    pass

main()