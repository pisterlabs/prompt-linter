import sys
import openai
openai.api_key = open("API.txt", "r").read()

text = open(sys.argv[1], 'r').read()

# prompt = """
# You are an expert programmer summarizing a git diff.
# Reminders about the git diff format:
# For every file, there are a few metadata lines, like (for example):
# ```
# diff --git a/lib/index.js b/lib/index.js
# index aadf691..bfef603 100644
# --- a/lib/index.js
# +++ b/lib/index.js
# ```
# This means that `lib/index.js` was modified in this commit. Note that this is only an example.
# Then there is a specifier of the lines that were modified.
# A line starting with `+` means it was added.
# A line that starting with `-` means that line was deleted.
# A line that starts with neither `+` nor `-` is code given for context and better understanding.
# It is not part of the diff.
# After the git diff of the first file, there will be an empty line, and then the git diff of the next file.

# Do not include the file name as another part of the comment.
# Do not use the characters `[` or `]` in the summary.
# Write every summary comment in a new line.
# Comments should be in a bullet point list, each line starting with a `-`.
# The summary should not include comments copied from the code.
# The output should be easily readable. When in doubt, write fewer comments and not more. Do not output comments that
# simply repeat the contents of the file.
# Readability is top priority. Write only the most important comments about the diff.

# EXAMPLE SUMMARY COMMENTS:
# ```
# - Raise the amount of returned recordings from `10` to `100`
# - Fix a typo in the github action name
# - Move the `octokit` initialization to a separate file
# - Add an OpenAI API for completions
# - Lower numeric tolerance for test files
# - Add 2 tests for the inclusive string split function
# ```
# Most commits will have less comments than this examples list.
# The last comment does not include the file names,
# because there were more than two relevant files in the hypothetical commit.
# Do not include parts of the example in your summary.
# It is given only as an example of appropriate comments.

# THE GIT DIFF TO BE SUMMARIZED: {}
# """.format(text)

prompt = """
You are an expert programmer summarizing a code change.
You went over every file that was changed in it.
For some of these files changes where too big and were omitted in the files diff summary.
Determine the best label for the commit.

Here are the labels you can choose from:

- build: Changes that affect the build system or external dependencies (example scopes: gulp, broccoli, npm)
- chore: Updating libraries, copyrights or other repo setting, includes updating dependencies.
- ci: Changes to our CI configuration files and scripts (example scopes: Travis, Circle, GitHub Actions)
- docs: Non-code changes, such as fixing typos or adding new documentation
- feat: a commit of the type feat introduces a new feature to the codebase
- fix: A commit of the type fix patches a bug in your codebase
- perf: A code change that improves performance
- refactor: A code change that neither fixes a bug nor adds a feature
- style: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
- test: Adding missing tests or correcting existing tests


THE FILE SUMMARIES:
###
{}
###

The label best describing this change:
""".format(text)

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
    	{"role": "user", "content": prompt},
    ]
)

print(response.choices[0].message.content)