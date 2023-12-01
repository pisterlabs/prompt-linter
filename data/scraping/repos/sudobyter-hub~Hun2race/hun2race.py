import argparse
import subprocess
from modules.engines.google_bard import GoogleBard
from modules.engines.openai_chatgpt import OpenaiChatgpt
from modules.documentors.latex import Latex
from modules.documentors.template import Template
from dotenv import load_dotenv
import os
load_dotenv()

# Set API keys
api_keys = {
    "bard": os.getenv("GOOGLE_BARD_API_KEY"),
    "open_ai": os.getenv("OPENAI_API_KEY")
}

#ASCII Art & Help Text
print('''
▒█░▒█ ▒█░▒█ ▒█▄░▒█ █▀█ ▒█▀▀█ ░█▀▀█ ▒█▀▀█ ▒█▀▀▀ 
▒█▀▀█ ▒█░▒█ ▒█▒█▒█ ░▄▀ ▒█▄▄▀ ▒█▄▄█ ▒█░░░ ▒█▀▀▀ 
▒█░▒█ ░▀▄▄▀ ▒█░░▀█ █▄▄ ▒█░▒█ ▒█░▒█ ▒█▄▄█ ▒█▄▄▄
By : sudobyter 
Usage : 
        -h for help 
        example : 
        python3 hun2race.py -f bug_bounty -v IDOR -t attacker.com -P "Found IDOR on the following domain etc..." -e bard/chatgpt -i <url_img1> <url_img2>
      ''')



# LaTeX Template with Dynamic Image Placeholder
BUG_BOUNTY_TEMPLATE = r"""
\documentclass{{article}}
\usepackage{{graphicx}}
\begin{{document}}
\title{{Bug Bounty Report}}
\author{{Security Researcher}}
\date{{\today}}
\maketitle
\newpage 
\section{{Summary}}
Format: Bug Bounty \\
Vulnerability Type: {vulnerability} \\
Host: {host}
\newpage 
\section{{Vulnerability Description}}
{vulnerability_desc}
\section{{Proof of Concept}}
{proof_of_concept}
\section{{Impact}}
{impact_description}
\section{{Recommendations}}
{suggestions}
\section{{Attachments}}
{image_latex}
\end{{document}}
"""


def main():
    parser = argparse.ArgumentParser(description='Security Research Report Generator')
    parser.add_argument('-uc', '--use-case', choices=Template().get_available_templates(), required=True, help='Report format')
    parser.add_argument('-v', '--vulnerability', required=True, help='Type of vulnerability')
    parser.add_argument('-t', '--target', required=True, help='Host where the vulnerability was found')
    parser.add_argument('-P', '--poc', help='PoC of the bug')
    parser.add_argument('-Pf', '--poc-file', help='File containing the PoC of the bug')
    parser.add_argument('-e', '--engine', choices=['bard', 'chatgpt'], required=True, help='Choice of description engine')
    parser.add_argument('-mu', '--images-urls', nargs='*', help='URLs of the images to include in the report', default=[])

    args = parser.parse_args()
    # Validate that either -P or -Pf is provided, but not both
    if not args.poc and not args.poc_file:
        print("Error: Either PoC (-P) or PoC file (-Pf) must be provided.")
        return
    elif args.poc and args.poc_file:
        print("Error: Both PoC (-P) and PoC file (-Pf) cannot be provided at the same time.")
        return

    # If PoC is provided via a file, read the content of the file
    if args.poc_file:
        with open(args.poc_file, 'r') as f:
            poc_content = f.read()
    else:
        poc_content = args.poc

    # Use either bard or chatgpt based on the user's choice
    if args.engine == 'bard':
        bard = GoogleBard(args.vulnerability, api_keys['bard'])
        vulnerability_desc, impact_description, suggestions = bard.get_contents()

    elif args.engine == 'chatgpt':
        chatgpt = OpenaiChatgpt(args.vulnerability, api_keys['open_ai'])
        vulnerability_desc, impact_description, suggestions = chatgpt.get_contents()

    latex = Latex(args.use_case, args.vulnerability, args.target, vulnerability_desc, poc_content, impact_description, suggestions, args.images_urls)

    latex_report = latex.generate_report()

    with open('security_report.tex', 'w') as f:
        f.write(latex_report)

    print("LaTeX report generated successfully.")

    # Compile LaTeX report to PDF and display any errors
    try:
        result = subprocess.run(['pdflatex', "security_report.tex"], capture_output=True, text=True)
        if result.returncode != 0:
            print("Error occurred while compiling LaTeX to PDF.")
            print(result.stderr)
        else:
            print("PDF report generated successfully.")
    except Exception as e:
        print(f"Error during LaTeX compilation: {e}")


if __name__ == '__main__':
    main()
