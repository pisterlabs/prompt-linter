#!/usr/bin/env python3
import sys
import requests
import argparse
import json
import openai
import os
import base64

SAMPLE_PROMPT = """
Write a pull request description focusing on the motivation behind the change and why it improves the project.
Go straight to the point.

The title of the pull request is "Enable valgrind on CI" and the following changes took place: 

Changes in file .github/workflows/build-ut-coverage.yml: @@ -24,6 +24,7 @@ jobs:
         run: |
           sudo apt-get update
           sudo apt-get install -y lcov
+          sudo apt-get install -y valgrind
           sudo apt-get install -y ${{ matrix.compiler.cc }}
           sudo apt-get install -y ${{ matrix.compiler.cxx }}
       - name: Checkout repository
@@ -48,3 +49,7 @@ jobs:
         with:
           files: coverage.info
           fail_ci_if_error: true
+      - name: Run valgrind
+        run: |
+          valgrind --tool=memcheck --leak-check=full --leak-resolution=med \
+            --track-origins=yes --vgdb=no --error-exitcode=1 ${build_dir}/test/command_parser_test
Changes in file test/CommandParserTest.cpp: @@ -566,7 +566,7 @@ TEST(CommandParserTest, ParsedCommandImpl_WhenArgumentIsSupportedNumericTypeWill
     unsigned long long expectedUnsignedLongLong { std::numeric_limits<unsigned long long>::max() };
     float expectedFloat { -164223.123f }; // std::to_string does not play well with floating point min()
     double expectedDouble { std::numeric_limits<double>::max() };
-    long double expectedLongDouble { std::numeric_limits<long double>::max() };
+    long double expectedLongDouble { 123455678912349.1245678912349L };
 
     auto command = UnparsedCommand::create(expectedCommand, "dummyDescription"s)
                        .withArgs<int, long, unsigned long, long long, unsigned long long, float, double, long double>();
"""

GOOD_SAMPLE_RESPONSE = """
## Description
The changes in this pull request focus on integrating Valgrind into our CI build process, which was not present before. Valgrind is a critical tool for detecting memory errors and ensuring the overall integrity of our project. By including Valgrind in the CI build, it provides us with immediate detection and reporting capabilities for any memory errors that might occur. This integration will help us in preventing undetected memory errors from these tests ending up in the production builds, thereby contributing to improving the project's overall quality.

## How Has This Been Tested?
The new changes were tested in the CI environment where Valgrind was successfully installed and run over the existing codes. Memory leak tests were a focus of the testing process. The new code and related elements consistently passed these tests indicating that the proposed inclusion of Valgrind is executing as expected.

## Deployment Plan
<!--- Describe deployment steps if any, e.g. setting new environment variables, N/A otherwise -->

## Screenshots:
<!--- Add screenshots, N/A otherwise -->
"""


def main():
    parser = argparse.ArgumentParser(
        description="Use ChatGPT to generate a description for a pull request."
    )
    parser.add_argument(
        "--github-api-url", type=str, required=True, help="The GitHub API URL"
    )
    parser.add_argument(
        "--github-repository", type=str, required=True, help="The GitHub repository"
    )
    parser.add_argument(
        "--pull-request-id",
        type=int,
        required=True,
        help="The pull request ID",
    )
    parser.add_argument(
        "--github-token",
        type=str,
        required=True,
        help="The GitHub token",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        required=True,
        help="The OpenAI API key",
    )
    parser.add_argument(
        "--jira-username",
        type=str,
        required=True,
        help="Jira username",
    )
    parser.add_argument(
        "--jira-api-token",
        type=str,
        required=True,
        help="Jira API token",
    )
    parser.add_argument(
        "--jira-issue-key",
        type=str,
        required=True,
        help="Jira issue key",
    )
    parser.add_argument(
        "--jira-api-url",
        type=str,
        required=True,
        help="Jira API URL",
    )
    parser.add_argument(
        "--allowed-users",
        type=str,
        required=False,
        help="A comma-separated list of GitHub usernames that are allowed to trigger the action, empty or missing means all users are allowed",
    )

    args = parser.parse_args()

    github_api_url = args.github_api_url
    repo = args.github_repository
    github_token = args.github_token
    pull_request_id = args.pull_request_id
    openai_api_key = args.openai_api_key
    jira_username = args.jira_username
    jira_api_token = args.jira_api_token
    jira_issue_key = args.jira_issue_key
    jira_api_url = args.jira_api_url

    allowed_users = os.environ.get("INPUT_ALLOWED_USERS", "")
    if allowed_users:
        allowed_users = allowed_users.split(",")
    open_ai_model = os.environ.get("INPUT_OPENAI_MODEL", "gpt-3.5-turbo")
    max_prompt_tokens = int(os.environ.get("INPUT_MAX_TOKENS", "1000"))
    model_temperature = float(os.environ.get("INPUT_TEMPERATURE", "0.6"))
    model_sample_prompt = os.environ.get(
        "INPUT_MODEL_SAMPLE_PROMPT", SAMPLE_PROMPT)
    model_sample_response = os.environ.get(
        "INPUT_MODEL_SAMPLE_RESPONSE", GOOD_SAMPLE_RESPONSE
    )
    authorization_header = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": "token %s" % github_token,
    }

    pull_request_url = f"{github_api_url}/repos/{repo}/pulls/{pull_request_id}"
    pull_request_result = requests.get(
        pull_request_url,
        headers=authorization_header,
    )
    if pull_request_result.status_code != requests.codes.ok:
        print(
            "Request to get pull request data failed: "
            + str(pull_request_result.status_code)
        )
        return 1
    pull_request_data = json.loads(pull_request_result.text)

    if pull_request_data["body"]:
        print("Pull request already has a description, skipping")
        return 0

    if allowed_users:
        pr_author = pull_request_data["user"]["login"]
        if pr_author not in allowed_users:
            print(
                f"Pull request author {pr_author} is not allowed to trigger this action"
            )
            return 0

    pull_request_title = pull_request_data["title"]

    pull_request_files = []
    # Request a maximum of 10 pages (300 files)
    for page_num in range(1, 11):
        pull_files_url = f"{pull_request_url}/files?page={page_num}&per_page=30"
        pull_files_result = requests.get(
            pull_files_url,
            headers=authorization_header,
        )

        if pull_files_result.status_code != requests.codes.ok:
            print(
                "Request to get list of files failed with error code: "
                + str(pull_files_result.status_code)
            )
            return 1

        pull_files_chunk = json.loads(pull_files_result.text)

        if len(pull_files_chunk) == 0:
            break

        pull_request_files.extend(pull_files_chunk)

        # Fetch the Jira issue description
        credentials = f'{jira_username}:{jira_api_token}'
        credentials_base64 = base64.b64encode(credentials.encode()).decode()

        # Create the headers with basic authentication using the API token
        headers = {
            'Authorization': f'Basic {credentials_base64}',
            'Content-Type': 'application/json',
        }

        # Construct the URL for the Jira issue
        url = f'{jira_api_url}/3/issue/{jira_issue_key}'

        # Send a GET request to retrieve the issue details
        response = requests.get(url, headers=headers)

        print(
            f'Jira issue description request status code: {response.status_code}')

        if response.status_code == 200:
            issue_data = response.json()
            description = ""

            if 'fields' in issue_data and 'description' in issue_data['fields']:
                description_data = issue_data['fields']['description']

                if 'content' in description_data:
                    for content in description_data['content']:
                        if content['type'] == 'paragraph':
                            for paragraph_content in content['content']:
                                if paragraph_content['type'] == 'text':
                                    description += paragraph_content['text'] + " "

            task_description = description.strip()  # Print the description
        else:
            print(
                f"Failed to fetch Jira issue description. Response: {response.text}")
            task_description = ""

        # Define an array of filenames to exclude
        exclude_filenames = ["package-lock.json"]

        completion_prompt = f"""
Take a deep breath and work on this problem step-by-step. 
Write a concise pull request description focusing on the motivation behind the change so that it is helpful for the reviewer to understand.
Go straight to the point, avoid verbosity.
Your pull request description should consist of four sections:
a. Description: This is the core of your description. 
b. How Has This Been Tested?: If you're uncertain, leave this section empty.
c. Deployment Plan: Leave this section blank for the author to fill out. 
d. Screenshots: Similar to the Deployment Plan, leave this section blank. 

Below is additional context regarding task for the changes made. Use them to better describe changes that took place: 
{task_description}

The title of the pull request is "{pull_request_title}" and the following changes took place: \n
"""
    for pull_request_file in pull_request_files:
        # Not all PR file metadata entries may contain a patch section
        # For example, entries related to removed binary files may not contain it
        if "patch" not in pull_request_file:
            continue

        filename = pull_request_file["filename"]

        if filename in exclude_filenames:
            continue

        patch = pull_request_file["patch"]
        completion_prompt += f"Changes in file {filename}: {patch}\n"

    max_allowed_tokens = 2048  # 4096 is the maximum allowed by OpenAI for GPT-3.5
    characters_per_token = 4  # The average number of characters per token
    max_allowed_characters = max_allowed_tokens * characters_per_token
    if len(completion_prompt) > max_allowed_characters:
        completion_prompt = completion_prompt[:max_allowed_characters]

    print(f"Using model: '{open_ai_model}'")

    openai.api_key = openai_api_key
    openai_response = openai.ChatCompletion.create(
        model=open_ai_model,
        messages=[
            {
                "role": "system",
                "content": "You are a world class expert full stack web developer having experience with nodejs, typescript, express who writes pull request descriptions adding 'description' and 'how has this been tested' sections.",
            },
            {"role": "user", "content": model_sample_prompt},
            {"role": "assistant", "content": model_sample_response},
            {"role": "user", "content": completion_prompt},
        ],
        temperature=model_temperature,
        max_tokens=max_prompt_tokens,
    )

    generated_pr_description = openai_response.choices[0].message.content
    redundant_prefix = "This pull request "
    if generated_pr_description.startswith(redundant_prefix):
        generated_pr_description = generated_pr_description[len(
            redundant_prefix):]
        generated_pr_description = (
            generated_pr_description[0].upper() + generated_pr_description[1:]
        )
    generated_pr_description = f'[{jira_issue_key}](https://portchain.atlassian.net/browse/{jira_issue_key})\n\n{generated_pr_description}'
    print(f"Generated pull request description: '{generated_pr_description}'")
    issues_url = "%s/repos/%s/issues/%s" % (
        github_api_url,
        repo,
        pull_request_id,
    )
    update_pr_description_result = requests.patch(
        issues_url,
        headers=authorization_header,
        json={"body": generated_pr_description},
    )

    if update_pr_description_result.status_code != requests.codes.ok:
        print(
            "Request to update pull request description failed: "
            + str(update_pr_description_result.status_code)
        )
        print("Response: " + update_pr_description_result.text)
        return 1


if __name__ == "__main__":
    sys.exit(main())
