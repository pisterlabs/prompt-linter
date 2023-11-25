import json
import os

from parsers import (
    PromptDetector,
    used_in_langchain_llm_call,
    used_in_openai_call,
    new_line_in_string,
    prompt_or_template_in_name,
)

detector = PromptDetector()

detector.add_heuristic(used_in_openai_call)
detector.add_heuristic(used_in_langchain_llm_call)
detector.add_heuristic(new_line_in_string)
detector.add_heuristic(prompt_or_template_in_name)

root_dir = "repos"

paths = []
for repo in os.listdir(root_dir):
    repo_path = os.path.join(root_dir, repo)
    for file in os.listdir(repo_path):
        file_path = os.path.join(repo_path, file)
        paths.append(file_path)

results = detector.detect_prompts(paths)

with open("prompts.json", "w") as file:
    json.dump(results, file)
