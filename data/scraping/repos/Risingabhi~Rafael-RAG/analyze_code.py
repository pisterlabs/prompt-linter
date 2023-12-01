#code fetch and analyse using openaiAPI, Langchain etc/

import os

import langchain

import openai


user_github = "user_github"

def get_repo(repo_queue):
	try:
		while not  repo_queue.empty():
			# Get the next repository from the queue
			repo = repo_queue.get()
			foldername = os.path.basename(repo)
			print(foldername)
			#find and print all files found in repo
			# Find and print all files found in repo
			path_folder = os.path.join(user_github, foldername)
			try:

				# Check if the directory exists before listing its files
				if os.path.exists(path_folder) and os.path.isdir(path_folder):
					list_of_files = os.listdir(path_folder)
					print("List of files:", list_of_files)
				else:
					print(f"Directory {path_folder} does not exist.")
			except Exception as e:
				print(f"An error occurred: {e}")
		return list_of_files


	except:
		pass

content_python_file =[]

def analysis_by_ai(problem_statement_user, list_of_files,repo_queue):
	"""construct a proper prompt to clarify user doubts\n
	1- find MAIN entry file from code.
	2- Explain if problem statement can be addressed by github repo"""
	repo = repo_queue.get()
	print("REPO",repo)
	foldername = os.path.basename(repo)
	path_folder = os.path.join(user_github, foldername)
	for files in list_of_files:
		sort_file = files.split('.')
		if sort_file[-1] == ".py":
			file_path = os.path.join(path_folder,files)
			
			with open(file_path, 'r') as f:
				content = f.read()
				print("content",content)
				content_python_file.append((files,content))
				return content_python_file


