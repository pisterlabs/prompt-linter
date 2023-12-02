import os
import logging
from openai import OpenAI
from dotenv import load_dotenv
from zurnai.split_file import split_file

load_dotenv()

client = OpenAI()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SYSTEM_PROMPT = """
Analyze and explain the vulnerabilities in this solidity code chunk, if there is any, other wise return "No vulnerabilities found".
Since the code is split into chunks, if you find a vulnerability, please explain how it can be exploited. If there is no vulnerabilities found in the code, just return "No vulnerabilities found".
No need to explain the code or why you can't find any vulnerabilities. Keep you explanation short and concise. 
Don't talk about the incomplete code, we alreade know that hte code given to you is just a chunk of the whole code.
""".strip()


def append_to_file(file_path, text):
    try:
        with open(file_path, "a+") as file:
            file.write(text)
    except:
        print("Exception occured while writing to file")


def write_report(file_path):
    whole_code, splits = split_file(file_path)
    logger.log(logging.INFO, "Split file into %s chunks", len(splits))
    if os.path.exists("report.md"):
        os.remove("report.md")
    logger.log(logging.INFO, "Report file is reset, writing report")
    file_name = file_path.split("/")[-1]
    append_to_file("report.md", f"# Report for {file_name}\n\n")
    append_to_file("report.md", "## Vulnerabilities found\n\n")
    for chunk in splits:
        logger.log(logging.INFO, "Writing report for chunk %s", chunk)
        append_to_file("report.md", "### Code chunk\n\n")
        append_to_file("report.md", f"```solidity\n{chunk}\n```\n\n")
        logger.log(logging.INFO, "Analyzing...")
        report = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": chunk},
            ],
        )
        append_to_file("report.md", "### Report\n\n")
        append_to_file("report.md", report.choices[0].message.content + "\n\n")

    append_to_file("report.md", "## Analysis as a whole\n\n")
    append_to_file("report.md", "### Code\n\n")
    append_to_file("report.md", f"```solidity\n{whole_code}\n```\n\n")
    logger.log(logging.INFO, "Analyzing...")
    report = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": whole_code},
        ],
    )
    append_to_file("report.md", "### Report\n\n")
    append_to_file("report.md", report.choices[0].message.content + "\n\n")
