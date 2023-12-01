import os, argparse
from translate_file import translate_and_save_file
from constants import str_to_language_code
from openai import OpenAI


def main():
    # parser setup
    parser = argparse.ArgumentParser(description='Translate markdown files')
    parser.add_argument('--replace', action='store_true', help='replace original files')
    parser.add_argument('--language',type=str, default='ja', help='language code to translate to')
    parser.add_argument('--extensions', nargs='+', type=str, default=['.md'], help='file extensions to translate')
    args = parser.parse_args()
    
    # set up OpenAI client
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
    # fetch target file paths
    with open("file_paths.txt", "r") as file:
        target_file_paths = file.read().splitlines()

    # Translate and replace markdown files
    for file_path in target_file_paths:
        translate_and_save_file(
            file_path, \
            dst_language_code=str_to_language_code(args.language), \
            client=client, \
            replace=args.replace
        )
        
if __name__ == "__main__":
    main()