import argparse
import subprocess
import os
import guidance
from pathlib import Path
import contextlib

gpt35turbo = guidance.llms.OpenAI("gpt-3.5-turbo-16k")

def get_excerpt(content):
  score = guidance("""
  {{#system~}}
  You generate two sentence summaries from markdown content.
  {{~/system}}

  {{#user~}}
  {{content}}
  Can you summarize this in two sentences?
  {{~/user}}

  {{#assistant~}}
  {{gen 'summary' max_tokens=100 temperature=0}}
  {{~/assistant}}
  """, llm=gpt35turbo)
  out = run_llm_program(score, content=content)
  return out['summary'].strip() 

def run_llm_program(program, *args, **kwargs):
    with open("log.txt", "a") as f, contextlib.redirect_stdout(
        f
    ), contextlib.redirect_stderr(f):
        return program(*args, **kwargs)

def add_excerpt_to_md_file(filename):

    with open(filename, 'r') as f:
        lines = f.readlines()

    excerpt_exists = False

    for i, line in enumerate(lines[1:], start=1):
        if line.strip().startswith('excerpt:'):
            excerpt_exists = True
        elif not excerpt_exists and line.strip() == "---":
            break

    if not excerpt_exists:
        # Generate the excerpt
        file_content = Path(filename).read_text()
        excerpt = get_excerpt(file_content)

        # Insert the excerpt
        lines.insert(i, f"excerpt: |\n    {excerpt}\n")

    with open(filename, 'w') as f:
        f.writelines(lines)

def main():
    parser = argparse.ArgumentParser(description='Add an excerpt to a markdown file.')
    parser.add_argument('--dir', help='The directory containing the markdown files.')
    parser.add_argument('--file', help='The path to a single markdown file.')

    args = parser.parse_args()

    if args.dir:
        # Process each markdown file in the directory
        for root, dirs, files in os.walk(args.dir):
            for file in files:
                if file.endswith('.md'):
                    path = os.path.join(root, file)
                    print(f"Starting: {path}")
                    add_excerpt_to_md_file(os.path.join(root, file))
                    print(f"Finishing: {path}")
    elif args.file:
        add_excerpt_to_md_file(args.file)
    else:
        print("Please provide either --dir or --file.")
        exit(1)

if __name__ == '__main__':
    main()
