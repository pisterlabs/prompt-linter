from github import Github 
from langchain.llms import OpenAI  
import os
import git
import re
from math import ceil
import sceret

def getrepo(username):
    g=Github(sceret.auth_token)
    repos  = g.get_user(username).get_repos()
    links = []
    for eachRepo in repos:
        link = f"{username}/{eachRepo.name}"
        links.append(link)
    return links


def findfiles(repositories):
    g= Github(sceret.auth_token)
    codes = {}
    files = []
    repoFiles = {}
    for each_repo in repositories:
        rfile = []
        repos = g.get_repo(each_repo)
        contents = repos.get_contents("")
        extensions = {'.py', '.c', '.cpp', '.java'}
        allfiles = []
        while contents:
            file_content = contents.pop(0)
            if file_content.type == "dir":
                 contents.extend(repos.get_contents(file_content.path))
            else:
                 allfiles.append(file_content.path)
        for file in allfiles:
            if file.endswith(tuple(extensions)):
                rfile.append(file)
                files.append(file)
                content = repos.get_contents(file)
                codes[file]=[content.decoded_content]
        repoFiles[each_repo] = [rfile]
    return repoFiles , codes


    
def complexityFinder(codes):
    score = {}
    llm = OpenAI(openai_api_key= sceret.OPENAI_API_KEY)
    for code in codes:
        sum = 0
        inputcode = codes[code]
        if len(inputcode[0]) > 3500:
            n = 3500
            chunks = [inputcode[0][i:i+n] for i in range(0, len(inputcode[0]), n)]
        else:
            chunks =inputcode
        for s in chunks: 
            querry = 'what is technical complexity of this code from 1 to 99 please provide a numerical digit as output:  \n '  
            Qcode = s
            querry += str(Qcode)
            out = llm.predict(querry)
            sc = re.findall(r'\d+', out)
            sum+=int(sc[0])
        if sum>100:
            reposcore = sum/ceil(len(inputcode[0])/3500)
        else:
            reposcore = sum
        score[code] = [reposcore]
    return score
        
def findResult(repo, scores):
    average_scores = {}

    for repo, files in repo.items():
        reposcores = []
        for file in files[0]:
            filescores = scores.get(file, [])
            reposcores.extend(filescores)
        
        if reposcores:
            average_score = sum(reposcores) / len(reposcores)
            average_scores[repo] = average_score

    return average_scores
