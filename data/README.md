# Data Collection

### Collecting repositories
Using GitHub's API, we have 1000 repositories based on the following parameters:

query = "langchain+OR+GUIDANCE+OR+LlamaIndex"
sort = "stars"
order = "asc"
per_page = 100  # Max 100
language = "python"


### Finding prompt files within the repositories
Using Github's search_code API, we then search through each repository for files 
that contain prompts or templates within .txt or .py files.

About 101 out of 1000 repositories contain prompt files.


### Collecting prompt files
We then collect URL for the raw file content of the prompt files for all the 101 repositories. 
This way, it will be easier to manually work with those files.

### Using gh api

`gh search code "from openai import OpenAI" --limit 100 --language "python" > res4.json`