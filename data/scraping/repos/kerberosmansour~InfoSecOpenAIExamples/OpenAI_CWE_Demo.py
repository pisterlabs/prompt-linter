import openai
import json
import os
from tqdm import tqdm

# Save the API Key as an enviromental vaieble (either in bash or ZSH or other) 
# export OPENAI_API_KEY="EnterKeyHERE"

openai.api_key = os.environ.get("OPENAI_API_KEY")

languages = [
    "Java",
    "Python",
    "Csharp",
    "Swift",
    "Kotlin",
    "PHP",
    "Rust",
    "Go",
    "JavaScript",
]

cwe_list = [
    'CWE-079: Improper Neutralization of Input During Web Page Generation (\'Cross-site Scripting\')',
    'CWE-020: Improper Input Validation',
    'CWE-078: Improper Neutralization of Special Elements used in an OS Command (\'OS Command Injection\')',
    'CWE-089: Improper Neutralization of Special Elements used in an SQL Command (\'SQL Injection\')',
    'CWE-022: Improper Limitation of a Pathname to a Restricted Directory (\'Path Traversal\')',
    'CWE-352: Cross-Site Request Forgery (CSRF)',
    'CWE-434: Unrestricted Upload of File with Dangerous Type',
    'CWE-306: Missing Authentication for Critical Function',
    'CWE-502: Deserialization of Untrusted Data',
    'CWE-287: Improper Authentication',
    'CWE-798: Use of Hard-coded Credentials',
    'CWE-276: Incorrect Default Permissions',
    'CWE-200: Exposure of Sensitive Information to an Unauthorized Actor',
    'CWE-522: Insufficiently Protected Credentials',
    'CWE-611: Improper Restriction of XML External Entity Reference',
    'CWE-918: Server-Side Request Forgery (SSRF)',
    'CWE-077: Improper Neutralization of Special Elements used in a Command (\'Command Injection\')',
    'CWE-295: Improper Certificate Validation',
    'CWE-094: Improper Control of Generation of Code (\'Code Injection\')',
    'CWE-269: Improper Privilege Management',
    'CWE-917: Improper Neutralization of Special Elements used in an Expression Language Statement (\'Expression Language Injection\')',
    'CWE-059: Improper Link Resolution Before File Access (\'Link Following\')',
    'CWE-319: Cleartext Transmission of Sensitive Information',
    'CWE-601: URL Redirection to Untrusted Site (\'Open Redirect\')',
    'CWE-532: Insertion of Sensitive Information into Log File'
]


def fetch_vulnerability_info(language, cwe, aspect):
    prompt = f"Language: {language}\nVulnerability: {cwe}\n\n{aspect}:"
    
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0.7,
    )
    
    return response.choices[0].text.strip()

output = {}

for language in tqdm(languages, desc="Languages"):
    output[language] = {}
    for cwe in tqdm(cwe_list, desc="CWEs", leave=False):
        output[language][cwe] = {
            "description": fetch_vulnerability_info(language, cwe, "Describe the vulnerability"),
            "vulnerable_code": fetch_vulnerability_info(language, cwe, "Show an example of vulnerable code in that language"),
            "remediation": fetch_vulnerability_info(language, cwe, "Show an example of remediation"),
            "semgrep_rule": fetch_vulnerability_info(language, cwe, "Create a Semgrep rule to detect the vulnerability"),
            "codeql_rule": fetch_vulnerability_info(language, cwe, "Create a CodeQL rule to detect the vulnerability"),
        }

with open("output.02.json", "w") as f:
    json.dump(output, f, indent=2)

def create_markdown(output):
    with open("output.02.md", "w") as f:
        for language in output:
            f.write(f"# {language}\n\n")
            for cwe in output[language]:
                f.write(f"## {cwe}\n\n")
                f.write("### Description\n\n")
                f.write(f"{output[language][cwe]['description']}\n\n")
                f.write("### Vulnerable Code\n\n")
                f.write(f"```{language.lower()}\n{output[language][cwe]['vulnerable_code']}\n```\n\n")
                f.write("### Remediation\n\n")
                f.write(f"```{language.lower()}\n{output[language][cwe]['remediation']}\n```\n\n")
                f.write("### Semgrep Rule\n\n")
                f.write(f"```yaml\n{output[language][cwe]['semgrep_rule']}\n```\n\n")
                f.write("### CodeQL Rule\n\n")
                f.write(f"```ql\n{output[language][cwe]['codeql_rule']}\n```\n\n")
            f.write("\n")

with open("output.02.json", "r") as f:
    output = json.load(f)

create_markdown(output)
