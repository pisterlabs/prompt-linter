from pathlib import Path
import re
import tiktoken
import guidance
import os
import subprocess

llm_string = "gpt-3.5-turbo-16k"
llm_model = guidance.llms.OpenAI(
    llm_string
)  # Make sure OPENAI_API_KEY is set in your environment variables
encoding = tiktoken.encoding_for_model(llm_string)

repos = ["https://github.com/webgpu/webgpu-samples", "https://github.com/gfx-rs/wgpu"]

for repo in repos:
    repo_name = repo.split("/")[-1].split(".git")[0]
    if not os.path.exists(repo_name):
        print(f"Cloning {repo} into {repo_name}")
        subprocess.run(
            ["git", "clone", "--depth", "1", "--single-branch", repo], check=True
        )

suffix = ".wgsl"
path = "./"
cache_file = "cache.md"
cache = ""

if not Path(cache_file).exists():
    wgsl_files = [
        (code, len(encoding.encode(code)))
        for code in [
            re.sub(
                r"^\s*\n",
                "",
                re.sub(r"//.*", "", open(file, "r").read()),
                flags=re.MULTILINE,
            )
            for file in Path(path).rglob(f"*{suffix}")
        ]
    ]
    wgsl_files.sort(key=lambda x: x[1])

    total_tokens = 0
    max_tokens = 14200

    with open(cache_file, "w") as md_file:
        md_file.write(
            "Use the syntax and style of following WGSL WebGPU Shading Language examples delimited by triple backticks to respond to user inputs.\n\n"
        )
        for code, token_count in wgsl_files:
            if total_tokens + token_count > max_tokens:
                break

            md_file.write("Example WGSL WebGPU Shading Language Code:\n")
            md_file.write("```wgsl\n")
            md_file.write(code.strip() + "\n")
            md_file.write("```\n\n")

            total_tokens += token_count

cache = open(cache_file, "r").read()


wgsl_bot = guidance(
    """
{{#system~}}
{{wgsl_cache}}
{{~/system}}

{{#user~}}
Respond to the following question according to the examples:
{{query}}
{{~/user}}

{{#assistant~}}
{{gen 'answer' temperature=0 max_tokens=1024}}
{{~/assistant}}
""",
    llm=llm_model,
)

query = input("Enter your query: ")

if not query.strip():
    print("User query is empty. Using default query.\n")
    query = "Write basic pixel code"

print(wgsl_bot(wgsl_cache=cache, query=query)["answer"])
