import base64
import json
import requests
import os
import openai

def parse_github():
    github_token = os.environ.get("GITHUB_TOKEN")
    org_name = os.environ.get('GITHUB_ORG')  # Replace with the GitHub organization name
    readme_dict = {}

    # Fetch repositories for the organization
    repo_url = f"https://api.github.com/orgs/{org_name}/repos?per_page=100"
    headers = {
        "Authorization": f"token {github_token}"
    }

    response = requests.get(repo_url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to fetch repositories: {response.json()}")
        return

    repos = response.json()

    # Loop through each repo and fetch README.md content
    repo_names = []
    for repo in repos:
        readme_url = f"https://api.github.com/repos/{org_name}/{repo['name']}/contents/README.md"
        response = requests.get(readme_url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to fetch README.md for {repo['name']}: {response.json()}")
            continue

        content = response.json()
        decoded_content = base64.b64decode(content['content']).decode('utf-8')
        readme_dict[repo['name']] = decoded_content
        repo_names.append(repo['name'])

    # Output the README contents as a dictionary
    print(f"Readme Contents: {readme_dict}")

    return repos, readme_dict, 

def consult_with_agi(messages, bug_description, repo_name, readme):
    # Define the API key and endpoint
    api_key = os.environ.get("OPENAI_API_KEY")

    # Prepare the prompt
    prompt = f"Given the bug description: {bug_description} and the repository README: {readme}, is it likely that this bug is contributed to by the specified repository? Explain."

    messages.append({"role": "user", "content": "I found a repository called {0} and its readme shows the following: {1}. can you tellme if the bug described above may be coming from here and if so, why?".format(repo_name, readme)})
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=messages)

    completion = completion.choices[0].message
    messages.append({"role": "system", "content": completion})
    print(completion)

def determine_buggy_repo(bug_description, repos, readmes):
    """Determine the most likely repo to contain the bug."""
    messages=[
        {"role": "system", "content": "You are a helpful software engineering intern helping debug customer complaints at a new machine learning startup called banana. Your job is to scan github readmes to figure out which repo is likely to contribute to a reported bug from a user. You'll read through the code in these repos, and propose imporvements that the team will review"},
        {"role": "user", "content": "I have the following bug report from a user: " + bug_description + " . And now I'm going to list out a bunch of github repo readmes and you're going to tell me which likely contributed to this bug. sound good?"},
        {"role": "system", "content": "sounds good, let's begin!"}]

    for repo in repos:
        readme = readmes.get(repo, None)
        if readme is None:
            print("README is None, skipping")
            continue

        # Define the API key and endpoint
        api_key = os.environ.get("OPENAI_API_KEY")

        # Prepare the prompt
        prompt = f"Given the bug description: {bug_description} and the repository README: {readme}, is it likely that this bug is contributed to by the specified repository? Explain."

        messages.append({"role": "user", "content": "I found a repository called {0} and its readme shows the following: {1}. can you tellme if the bug described above may be coming from here and if so, why?".format(repo, readme)})
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=messages)

        completion = completion.choices[0].message['content']
        messages.append({"role": "system", "content": completion})
        print(completion)

    messages.append({"role": "user", "content": "Ok so what is the repo name you think is most likely among all of these then?"})
    completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                messages=messages)
    print(completion)
    return completion.choices[0].message['content']


def ingest(repo_name):
    import os
    os.environ["OPENAI_API_KEY"] = "API here"

    from langchain.llms import OpenAI
    import git

    llm = OpenAI(model_name="text-davinci-003", n=1, best_of=5,max_tokens=-1)

    from langchain.vectorstores import Chroma
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.docstore.document import Document

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma("repo_embeddings", embeddings, persist_directory="repo_embeddings")
    repos = [""" list of github repos here"""]

    page_content = "lets see how this embeds"*100
    id = vectorstore.add_documents([Document(page_content=page_content)])
    import shutil

    try:
        shutil.rmtree('repo')
    except:
        pass
    doc_to_repo = {}
    id_to_repo = {}
    for repo in repos:
        print("Handling repo: ", repo)

        _ = git.Repo.clone_from(repo,'repo')

        list_of_files = []
        for filename in os.listdir('repo'):
            f = os.path.join('repo', filename)
            # checking if it is a file
            if os.path.isfile(f):
                if f.endswith((".mod",".sum",".gitignore",".md",".sh","Dockerfile")):
                    continue
                list_of_files.append(f)


        all_desc = ""
        for code_file in list_of_files:
            f = open(code_file, "r")
            code = f.read()
            f.close()
            prefix = "Describe what the following code does in detail-\n"
            #print(llm(prefix+code))
            try:
                all_desc += llm(prefix+code)

            except:
                #Hack to deal with long files
                if len(code)%2 == 0:
                    code1 = code[0:len(code)//2]
                    code2 = code[len(code)//2:]
                else:
                    code1 = code[0:(len(code)//2+1)]
                    code2 = code[(len(code)//2+1):]

                desc1 = llm(prefix+code1)
                desc2 = llm(prefix+code2)

                all_desc += desc1 + desc2




        repo_summary = "Individual code file in a repo perform following tasks, what does the entire service do in detail?\n" + all_desc
        print(llm(repo_summary))
        id = vectorstore.add_documents([Document(page_content=all_desc)])[0]
        doc_to_repo[all_desc] = repo
        id_to_repo[id] = repo
        shutil.rmtree('repo')

        print("ids to repo: ", id_to_repo)
        vectorstore.persist()
        res = vectorstore.similarity_search("my build is stuck/suddenly slow",k=1)
        print(f"Bug seems to be from {doc_to_repo[res[0].page_content]}")

if __name__ == "__main__":
    DEBUG = False

    if DEBUG:
        repos = ["pod-deployer", "commit-handler", "build-service", "backend", "autoscaler"]
        readmes = {"pod-deployer": "Scan for all running deployments in the cluster, keep them updated with model settings and add new deployments to the cluster when it shows up in the database after a completed build in build service.", "commit-handler": "Listens for new commits to users potassium apps, manages the pipeline of calling build service, pod-deployer, making sure the commit turns into working build.", "build-service": "builds useres potassium app on latest commit into docker image and pushes to our db so pod-deploy can deploy it into the cluster", "autoscaler": "autoscale deployments to handle more traffic", "backend": "crud app to handle requests made in our banana webapp"}
    else:
        repos, readmes, = parse_github()

    bug_description = "My app says its stuck in deploying state"

    likely_repo = determine_buggy_repo(bug_description, repos, readmes)

    print("Likely repo: ", likely_repo)
    ingest(likely_repo)
