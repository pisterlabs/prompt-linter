import cohere
import os
import time
from tqdm import tqdm
import configparser
import argparse
import pickle

def walkDirContent(path):
    # recursively walk through dir to get all markdown files
    # returns a dictionary where {filename1: content1, filename2: content2, ...}
    text_content = {}
    # only getting a subset of the directories and files because it's a lot
    for item in os.listdir(path):
        # print(item)
        content = os.path.join(path, item)
        if os.path.isdir(content):
            text_content.update(walkDirContent(content))
        elif content.endswith(".md"):
            with open(content) as fp:
                text_content[item] = " ".join(fp.readlines())
    return text_content

def get_cohere_api_key(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    return config.get('cohere', 'api_key')

def generate_summaries(co, content, call_limit = 5):
    calls = 0
    summaries = {}
    for key in tqdm(content.keys()):
        if calls == call_limit:
            time.sleep(60)
            calls = 0
        try:
            if len(content[key]) > 250:
                summaries[key] = co.summarize(content[key], model='summarize-medium', length='short', extractiveness='high').summary
            else:
                summaries[key] = content[key]
            calls += 1
        except:
            print("Error generating summary for file: {}".format(key))
    return summaries


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Generate summarizations for all files in a database")
    parser.add_argument("-i", "--input_dir", type=str, default="../my-second-brain", help="Input directory containing markdown files.")
    parser.add_argument("-c", "--config", type=str, default="../config.ini", help="Configuration file containing the API key.")
    parser.add_argument("-w", "--write_path", type=str, default="../Data", help="Directory where output should be saved")
    parser.add_argument("-n", "--name", type=str, default="cohere_summaries", help="Name of the file containing cohere summaries")
    args = parser.parse_args()
    
    input_dir  = args.input_dir
    config = args.config
    write_path = args.write_path
    name = args.name

    # Get API key and create cohere object
    api_key = get_cohere_api_key(config_path=config)
    co = cohere.Client(api_key)

    # Get all text content
    content = walkDirContent(input_dir)
    
    # Create summaries
    summaries = generate_summaries(co, content)

    # Save the summaries as a pickle file
    with open(write_path + "/" + name, "wb") as fp:
        pickle.dump(summaries, fp)





if __name__ == '__main__':
    main()