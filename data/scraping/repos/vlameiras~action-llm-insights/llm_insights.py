import os
import sys
import openai

# Set API key
openai.api_key = os.environ["OPENAI_API_KEY"]


def get_git_diff():
    try:
        # Get latest commit
        print("Getting latest git commit...")
        commit = os.popen("git rev-parse HEAD").read().strip()

        # Get latest git diff between staged files and HEAD
        print("Getting latest git diff...")
        diff = os.popen(f"git diff --cached {commit}").read()

        # Get latest git staged files
        print("Getting latest git staged files...")
        staged_files = os.popen("git diff --name-only --cached").read().splitlines()

        return diff, staged_files
    
    except Exception as e:
        print(e)
        sys.exit(1)
        

def get_code_insights(diff, file_contents):

    try:
        # Get code insights using OpenAI
        print("Getting code insights from OpenAI...")
        
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=(
                f"Analyze the following code diff and the contents "
                f"of the changed files. Provide potential optimizations, "
                f"observations, bugs detected, and suggested fixes. All "
                f"the insights should be detailed and relevant for the developer. "
                f"Present code snippets for bugs and fixes:\n\n"
                f"--- Code Diff ---\n{diff}\n\n"
                f"--- Changed Files ---\n{file_contents}\n\n"
                f"Use colors to pretty print the output and highlight each section\n"
            ),
            temperature=0.1,
            max_tokens=300,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        
        return response.choices[0].text.strip()

    except Exception as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    diff, file_contents = get_git_diff()
    if len(file_contents) == 0:
        print("No staged files found. Exiting...")
        sys.exit(0)
    insights = get_code_insights(diff, file_contents)
    print(insights)
