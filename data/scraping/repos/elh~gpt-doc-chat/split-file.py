import sys
import os
import argparse
from dotenv import load_dotenv
import openai

NUM_OF_LINES=20

def main():
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default="", help='file to split')
    parser.add_argument('--split-lines', type=int, default="", help='number of lines to split on')
    args = parser.parse_args()
    if args.file == "":
        print("ERROR: file is required")
        sys.exit(1)

    split_count = NUM_OF_LINES
    if args.split_lines > 0:
        split_count = args.split_lines

    with open(args.file) as fin:
        fout = open(args.file + "_split_0.txt","wb")
        for i,line in enumerate(fin):
            fout.write(bytes(line, 'utf-8'))
            if (i+1)%split_count == 0:
                fout.close()
                fout = open(args.file + "_split_%d.txt"%(i/split_count+1),"wb")
        fout.close()

if __name__ == "__main__":
    main()
