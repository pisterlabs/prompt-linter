# Importing modules
try:
    from github import Github
    import threading
    import time
    import os
    from colorama import Back, Style, Fore
    import openai
except ImportError as MissingModule:
    print("[-] An error occured while importing modules : {}".format(MissingModule))

class GitPullBot:
    def __init__(self):
        self.GITHUB_TOKEN = "" # Add you own Github token (settings => at the bottom "Developer Settings" then personal access token)
        self.GITHUB_API = Github(self.GITHUB_TOKEN) 
        self.processed_pulls_file = "processed_pulls.txt" # The file wich store the current processed pulls

    # Load the .txt file to check if the current pull request has already been replied
    def load_processed_pulls(self):
        try:
            with open(self.processed_pulls_file, "r") as file:
                return set(map(int, filter(None, file.read().splitlines())))
        except FileNotFoundError:
            return set()

    # If the current pull request hasn't been replied, so reply to it then add it to the .txt file
    def save_processed_pulls(self, processed_pulls):
        with open(self.processed_pulls_file, "w") as file:
            for pull_id in processed_pulls:
                file.write(str(pull_id) + "\n")

    # Retrieve all current opened pull requests on the current repository
    def OnGetOpenPullRequests(self, GitRepositoryName):
        repo = self.GITHUB_API.get_repo(GitRepositoryName)
        open_pull_requests = repo.get_pulls(state='open')
        return open_pull_requests

    # Generate the reply with the OpenAI API, feel free to change the prompt might be not adapted for your current project
    def OnGenerateCodeReviewComment(self, AICodeSnippet):
        # Your openAI API Key
        openai.api_key = ""
        # Default prompt, feel free to change it
        GitPullBotPrompt = f"You are a Github Pull Request Assistant Bot, This is a code snippet from a GitHub pull request. This code snippet has a potential issue:\n\n{AICodeSnippet}\n\n<b>Please provide the following</b>:\n\n- ⚙ <b>Potential issue</b>:\n\n- :bar_charts: <b>Level of the error</b> (Low/High/Critical):\n\n- :rocket: <b>How to fix it</b>: (Example code) "
        # Title Header
        TitleHeader = "This is an automatic message from <a href='https://github.com/devmalcolm/gitpullbot'>GitPullBot</a> :robot:\n\n" 
        # Warning header
        WarningSection = "⚠ <b>This message is for pre-checking pull requests and might get some incorrect suggestions</b>\n\n"
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", # Right now using the 3.5 turbo model, if you need another one, change it
                # Handling message / roles
                messages=[
                    {"role": "system", "content": GitPullBotPrompt},
                    {"role": "user", "content": AICodeSnippet}
                ],
                # Max token, the more you got, the more you consume data
                max_tokens=250
            )
            generated_comment = response.choices[0].message["content"].strip()
            TitleHeader += WarningSection
            TitleHeader += generated_comment
            return TitleHeader
        except Exception as e:
            print("An error occurred while generating comment:", e)
            return "GitPullBot encountered an error while generating a reply."
    
    # Once done, post the generated comment by the AI into the pull request
    def post_code_review_comment(self, pull_request, comment):
        pull_request.create_issue_comment(comment)

    # Handle the main function, check for pull request, then print information
    def OnExecuteGitPullBot(self, GitRepositoryName):
        processed_pulls = self.load_processed_pulls()
        open_pull_requests = self.OnGetOpenPullRequests(GitRepositoryName)

        for pull_request in open_pull_requests:
            pull_id = pull_request.number
            pull_title = pull_request.title
            if pull_id in processed_pulls:
                # This print line might spam a bit, in case you can just add a pass, then delete / comment this print line
                print(f"[{Fore.RED}-{Style.RESET_ALL}] Pull request {Fore.RED}#{pull_id}{Style.RESET_ALL} ({Fore.YELLOW}{pull_title}{Style.RESET_ALL}) has already been replied to.")
            else:
                return_code = pull_request.get_files()[0].patch
                comment = self.OnGenerateCodeReviewComment(return_code)
                self.post_code_review_comment(pull_request, comment)
                processed_pulls.add(pull_id)
                print(f"[{Fore.GREEN}*{Style.RESET_ALL}] Replied to new pull request {Fore.GREEN}#{pull_id}{Style.RESET_ALL} ({Fore.YELLOW}{pull_title}{Style.RESET_ALL}).")

        self.save_processed_pulls(processed_pulls)

# Thread that run each x times in the background in order to have the script h24 running, (In order to host on a VPS / Server for example)
def GitPullBotThread(repository_name):
    os.system("cls")
    print(f"[{Fore.GREEN}*{Style.RESET_ALL}] Starting {Fore.GREEN}GitPullBot{Style.RESET_ALL} GitHub Integration...")
    time.sleep(2)
    x = GitPullBot()
    GitPullBot_Timeout_Checking = 60 * 15 # Frequency bot will check for new pull requests, by default i set it to 15 minutes. Feel free to change it (Minimum: 15s)
    print(f"[{Fore.GREEN}*{Style.RESET_ALL}] GitPullBot Started.\n\n")
    print("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
    print("┃                 GitPullBot Log               ┃")                   
    print("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n")
    while True:
        x.OnExecuteGitPullBot(repository_name)
        time.sleep(GitPullBot_Timeout_Checking)

if __name__ == "__main__":
    repository_name = "devmalcolm/gitpullbot" # Modify with your user/repo_name 
    
    # Start the bot thread
    GitPullBotThread = threading.Thread(target=GitPullBotThread, args=(repository_name,))
    GitPullBotThread.daemon = True  # Allow the script to exit even if the thread is running
    GitPullBotThread.start()

    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")
