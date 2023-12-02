"""make file descrips for better lookup of relevant files"""

import os
import json
import logging as log
from collections import defaultdict
from threading import Lock

from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]


# def complete(prompt):
#     wompwomp = "WOMPWOMP: " + prompt + " WOMP"
#     return wompwomp


def complete(prompt: str):
    bigger_prompt = f"""{HUMAN_PROMPT}

{prompt}

{AI_PROMPT}
  """

    anthropic = Anthropic()
    completion = anthropic.completions.create(
        model="claude-instant-1", max_tokens_to_sample=3000, prompt=bigger_prompt
    )

    return completion.completion


def generate_file_descrips(paths, org, name, repo_root):
    """what"""
    filter_filepaths(paths)
    log.debug("Generating descriptions for filtered_paths.")
    pierre = ThankYouPierre(org, name, repo_root)
    descriptions = pierre.get_descriptions()

    return descriptions


def filter_filepaths(paths):
    """filter out non-relevant files"""

    # Filter out file suffixes
    bad_suffixes = ['.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg', '.ttf', '.woff', '.woff2', '.eot', '.mp4', '.mp3',
                    '.wav', '.ogg', '.webm', '.zip', '.tar', '.gz', '.rar', '.7z', '.pdf', '.doc', '.docx', '.xls',
                    '.xlsx', '.ppt', '.pptx', '.exe', '.dll', '.so', '.a', '.o', '.obj', '.lib', '.pyc', '.pyo',
                    '.class', '.jar', '.war', '.ear', '.iml', '.idea', '.DS_Store', '.gitignore', '.gitattributes',
                    '.gitmodules', '.gitkeep', '.git', '.hgignore', '.hg', '.svn', '.cvs', '.bzrignore', '.bzr',
                    '.npmignore', '.npmrc', '.yarnrc', '.yarnclean', '.yarn-integrity', '.yarnclean',
                    '.yarn-metadata.json', '.yarn-tarball.tgz', '.yarncl']

    # Filter out file prefixes
    bad_prefixes = ['ios', 'android', '.git', '.hg', '.svn', '.cvs', '.bzr', '.npm', '.yarn']

    # Use list comprehension to filter out bad suffixes and prefixes
    filtered_paths = [path for path in paths if not any(path.endswith(suffix) for suffix in bad_suffixes) and not any(
        path.startswith(prefix) for prefix in bad_prefixes)]

    return filtered_paths


mutexes = defaultdict(lambda: Lock())


class ThankYouPierre:
    extensions = (
        '.js', '.jsx', '.py', '.json', '.html', '.css', '.scss', '.yml', '.yaml', '.ts', '.tsx', '.ipynb', '.c', '.cc',
        '.cpp', '.go', '.h', '.hpp', '.java', '.sol', '.sh', '.txt', '.c', '.hpp', '.h', '.pdf', 'docx', '.sh', '.cfg',
        '.md')
    directory_blacklist = ('build', 'dist', '.github', 'site', 'tests', "node_modules")

    def __init__(self, org, name, local_path):
        self.org = org
        self.canon_name = org + "." + name

        self.local_path = local_path
        self.tmp_path = local_path + "/.pierre"
        os.makedirs(self.tmp_path, exist_ok=True)
        self.num_files = 0
        self.mutex = mutexes[self.canon_name]

    def walk(self, max_num_files=1000):
        num_files = 0
        for root, dirs, files in os.walk(self.local_path, topdown=True):
            files = [f for f in files if not f[0] == '.' and f.endswith(ThankYouPierre.extensions)]
            dirs[:] = [d for d in dirs if d[0] != '.' and not d.startswith(ThankYouPierre.directory_blacklist)]
            for name in files:
                filename = os.path.join(root, name)

                try:
                    with open(filename, 'r') as f:
                        code = f.read()
                except UnicodeDecodeError:
                    continue

                if code.strip() == '': continue

                if num_files >= max_num_files:
                    return

                yield filename, root, dirs, code

                num_files += 1
                self.num_files = num_files

    def load_descriptions(self):
        save_path = os.path.join(self.tmp_path, f"{self.canon_name}_descriptions.json")
        if not os.path.exists(save_path):
            return None
        with open(save_path, 'rb') as f:
            data = json.load(f)
        return data

    def get_descriptions(self, save=True, save_every=10):
        with self.mutex:
            descriptions = self.load_descriptions()
            if descriptions is not None and len(descriptions) == self.num_files:
                return descriptions

            if descriptions is None:
                descriptions = {}

            generator = self.walk()
            description_prompt = 'A 1-sentence summary in plain English of the above code, with no other commentary, is:'
            num_files = len(descriptions)
            for filename, root, dirs, code in generator:
                # Skip files that already have descriptions
                if filename in descriptions:
                    continue
                extension = filename.split('.')[-1]
                if len(code) / 3 > 100000:
                    code = code[0:300000]
                prompt = f'File: {filename}\n\nCode:\n\n```{extension}\n{code}```\n\n{description_prompt}\nThis file'
                try:
                    description = complete(prompt)
                except Exception:
                    log.exception("Error doing completion for :%s", filename)
                    continue
                descriptions[filename] = description

                log.debug(f"{filename}: {description}")

                if save and (num_files % save_every == 0):
                    log.debug(f'Saving descriptions for {num_files} files')
                    self.save_descriptions(descriptions)

                num_files += 1

            if save:
                self.save_descriptions(descriptions)

            return descriptions

    def save_descriptions(self, descriptions):
        save_path = os.path.join(self.tmp_path, f"{self.canon_name}_descriptions.json")
        with open(save_path, 'w') as f:
            json.dump(descriptions, f)
