#!/usr/bin/python3
"""
Cheers https://github.com/elastic/chatgpt-log-analysis/blob/main/app.py
"""
import os
import sys
import openai
import requests
import argparse
from copr.v3 import Client

from pprint import pprint

openai.api_key = os.environ.get("OPEN_API_TOKEN", None)


# logs contain a bunch of lines that are present in all logs and really provide any value
# strip them out
# there is a possibility these could add context but at the same time will make queries lighter
FILTER_THESE_OUT = (
    "Copr build error: Build failed\n",
    "INFO: Results and/or logs in: /var/lib/copr-rpmbuild/results\n",
    "INFO: Cleaning up build root ('cleanup_on_failure=True')\n",
    "Start: clean chroot\n",
    "INFO: unmounting tmpfs.\n",
    "Finish: clean chroot\n",
    "No matches found for the following disable plugin patterns: local, spacewalk, versionlock\n",
    # " # /usr/bin/systemd-nspawn -q -M 14eada62b44..."  # not sure about this one, it's long but has a lot of info in
    "Updating Subscription Management repositories.\n",
    "Unable to read consumer identity\n",
    "This system is not registered with an entitlement server. You can use subscription-manager to register.\n",
)

# Prompt GPT to analyze the logs
USER_PROMPT = (
        "The RPM build process failed and you need to fix it. " +
        "Please review the logs messages below, explain the root cause for the error and "
        "how it should fixed in the most optimal way. " +
        "The logs start here.\n"
)
# configure the AI model's role so we get proper answers
SYSTEM_PROMPT = "You are an RPM Package Maintainer and an upstream developer." + \
    " You are responsible for RPM builds to successfully complete."


def get_build_logs(build_id):
    """ provide build logs for given failed Copr build as text """
    client = Client.create_from_config_file()

    build = client.build_proxy.get(build_id)

    owner = build['ownername']
    project_name = build['projectname']
    pkg = build['source_package']['name']

    if build['source_package']['name'] is None:
        # srpm failed
        chroot = "srpm-builds"
    else:
        for chroot in build['chroots']:
            build_chroot = client.build_chroot_proxy.get(build_id, chroot)
            if build_chroot['state'] == 'failed':
                break
        else:
            return None

    pkg = "" if chroot == "srpm-builds" else f"-{pkg}"
    logs_url = (
        "https://download.copr.fedorainfracloud.org/"
        f"results/{owner}/{project_name}/{chroot}/"
        f"{build_id:08d}{pkg}/builder-live.log"
    )
    return requests.get(logs_url).text


def get_logs_snippet(logs):
    """
    As LLMs limit input size, we need to strip the logs from useless information.

    We filter out lines that are static and don't provide any specific info.

    :param logs: logs as text
    :return: subset of the logs as text
    """
    # only work with last 4k where the errors typically are
    tail_logs = logs[-4096:]
    for dupe in FILTER_THESE_OUT:
        tail_logs = tail_logs.replace(dupe, "")
    new_len = len(tail_logs)
    if new_len < 4096:
        print(f"\nWe have saved {4096 - new_len} characters.\n")
    return tail_logs


def prompt_gpt(build_id: int, dry_run: bool):
    # OpenAI's docs
    # What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random,
    # while lower values like 0.2 will make it more focused and deterministic.
    # We generally recommend altering this or top_p but not both.
    temperature = 0.5  # defaults to 1

    # OpenAI's docs
    # An alternative to sampling with temperature, called nucleus sampling, where the model considers the results
    # of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability
    # mass are considered. We generally recommend altering this or temperature but not both.
    top_p = 1  # defaults to 1

    full_logs = get_build_logs(build_id)
    tail_logs = get_logs_snippet(full_logs)
    print("These are the logs to send that we send to the AI model")
    print(f"```\n{tail_logs}\n```")
    if dry_run:
        return f"# Merge the logs above with this prompt:\n{SYSTEM_PROMPT} {USER_PROMPT}"
    else:
        # TODO: trick GPT to output JSON and process it
        # "Output in this JSON format: {\"short_summary\": \"<TBD>\", \"steps_to_fix\": [\"<step1>\", \"step2>\"]}. " +
        analysis_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # TODO we want GPT4!!!!!!
            messages= [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT + tail_logs}
            ],
            temperature=temperature,
            top_p=top_p,
        )
        pprint(analysis_response)
        return analysis_response["choices"][0]["message"]["content"]


def main():
    parser = argparse.ArgumentParser(
        prog='my-rpm-build-broke',
        description='My RPM Build Broke :(',
        epilog='Please help me analyze and resolve the build failure')
    parser.add_argument('copr_build_id', help="Failed Copr Build ID to analyze", type=int)
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help="Do NOT query OpenAI, only obtain the logs and present the prompt.")
    args = parser.parse_args()

    out = prompt_gpt(build_id=args.copr_build_id, dry_run=args.dry_run)

    print(out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
