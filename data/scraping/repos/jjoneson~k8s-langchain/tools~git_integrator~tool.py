from pydantic import BaseModel
from langchain.tools.base import BaseTool
import pygit2

class GitModel(BaseModel):
    username: str
    password: str

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_credentials(cls, username: str, password: str):
        return cls(username=username, password=password)
    
    def clone_repository(self, remote_repository_url: str) -> str:
        try:
            # local repository path should be everything after the first slash after the first double slash
            # example remote_repository_url: https://github.com/jjoneson/langchain.git
            # example local_repository_path: jjoneson/langchain
            local_repository_path = remote_repository_url.split('//')[1].split('/')[1:]
            local_repository_path = '/'.join(local_repository_path)
            remote_repository_url = remote_repository_url.replace('https://', f'https://{self.username}:{self.password}@')
            pygit2.clone_repository(remote_repository_url, local_repository_path)
            return f'Repository cloned to {local_repository_path}'
        except Exception as e:
            return f'Error cloning repository: {e}'
        
    def open_repository(self, local_repository_path: str) -> pygit2.Repository:
        try:
            return pygit2.Repository(local_repository_path)
        except Exception as e:
            return f'Error opening repository: {e}'

    def write_file(self, local_repository_path: str, file_path: str, file_content: str) -> str:
        try:
            with open(local_repository_path + "/" + file_path, 'w') as f:
                f.write(file_content)
            return f'File {file_path} written successfully'
        except Exception as e:
            return f'Error writing file: {e}'

    
    def commit(self, local_repository_path: str, message: str) -> str:
        try:
            repository = self.open_repository(local_repository_path)
            repository.index.add_all()
            repository.create_commit('HEAD', repository.default_signature, repository.default_signature, message, repository.index.write_tree(), [])
            return f'Commit {message} created successfully'
        except Exception as e:
            return f'Error creating commit: {e}'
    
    def push(self, local_repository_path: str) -> str:
        try:
            repository = self.open_repository(local_repository_path)
            repository.remotes[0].push([repository.head.name])
            return f'Pushed to {repository.remotes[0].url} successfully'
        except Exception as e:
            return f'Error pushing: {e}'
    
    def create_branch(self, local_repository_path: str, branch_name: str) -> str:
        try:
            repository = self.open_repository(local_repository_path)
            index = repository.index
            tree = index.write_tree()
            repository.create_reference_direct('refs/heads/' + branch_name, tree, False)
            repository.checkout('refs/heads/' + branch_name)
            return f'Branch {branch_name} created successfully'
        except Exception as e:
            return f'Error creating branch: {e}'
    
    def checkout_branch(self, local_repository_path: str, branch_name: str) -> str:
        try:
            repository = self.open_repository(local_repository_path)
            branch = repository.lookup_branch(branch_name)
            ref = repository.lookup_reference(branch.name)
            repository.checkout(ref)
            return f'Checked out branch {branch_name} successfully'
        except Exception as e:
            return f'Error checking out branch: {e}'
    
    def pull(self, local_repository_path: str) -> str:
        try:
            repository = self.open_repository(local_repository_path)
            repository.remotes[0].fetch()
            repository.checkout_tree(repository.get(repository.head.target))
            return f'Pulled from {repository.remotes[0].url} successfully'
        except Exception as e:
            return f'Error pulling: {e}'
    
    def add_file(self, local_repository_path: str, file_path: str) -> str:
        try:
            repository = self.open_repository(local_repository_path)
            repository.index.add(file_path)
            return f'Added file {file_path} successfully'
        except Exception as e:
            return f'Error adding file: {e}'
    
    def get_file_content(self, local_repository_path: str, file_path: str) -> str:
        try:
            repository = self.open_repository(local_repository_path)
            return repository[file_path].data.decode('utf-8')
        except Exception as e:
            return f'Error getting file content: {e}'
    

class GitRepositoryCloneTool(BaseTool):
    """Tool for cloning a git repository"""
    name = 'git_repository_clone'
    description = """
    Can be used to clone a git repository
    Input should be an https url to the repository, i.e. https://github.com/jjoneson/k8s-explorer.git
    Returns a string containing the local path to the repository
    """
    model: GitModel

    def _run(self, tool_input: str) -> str:
        """Run the tool"""
        return self.model.clone_repository(tool_input)

    async def _arun(self, tool_input: str) -> str:
        """Run the tool"""
        return self._run(tool_input)
    
class GitRepositoryCreateBranchTool(BaseTool):
    """Tool for creating a git branch"""
    name = 'git_repository_create_branch'
    description = """
    Can be used to create a git branch
    Input should be a comma separated list containing the local repository path followed the name of the branch to create
    Returns a string containing the name of the branch that was created
    """
    model: GitModel

    def _run(self, tool_input: str) -> str:
        """Run the tool"""
        local_repository_path, branch_name = tool_input.split(',', 1)
        # clean spaces
        local_repository_path = local_repository_path.strip()
        branch_name = branch_name.strip()
        return self.model.create_branch(local_repository_path, branch_name)

    async def _arun(self, tool_input: str) -> str:
        """Run the tool"""
        return self._run(tool_input)

class GitRepositoryCheckoutBranchTool(BaseTool):
    """Tool for checking out a git branch"""
    name = 'git_repository_checkout_branch'
    description = """
    Can be used to checkout a git branch
    Input should be a comma separated list containing the local repository path followed by the name of the branch to checkout
    Returns a string containing the name of the branch that was checked out
    """
    model: GitModel

    def _run(self, tool_input: str) -> str:
        """Run the tool"""
        local_repository_path, branch_name = tool_input.split(',', 1)
        # clean spaces
        local_repository_path = local_repository_path.strip()
        branch_name = branch_name.strip()
        return self.model.checkout_branch(local_repository_path, branch_name)

    async def _arun(self, tool_input: str) -> str:
        """Run the tool"""
        return self._run(tool_input)
    
class GitRepositoryPullTool(BaseTool):
    """Tool for pulling from a git repository"""
    name = 'git_repository_pull'
    description = """
    Can be used to pull from a git repository
    Input should be the local repository path
    Returns a string containing the result of the pull
    """
    model: GitModel

    def _run(self, tool_input: str) -> str:
        """Run the tool"""
        return self.model.pull(tool_input)

    async def _arun(self, tool_input: str) -> str:
        """Run the tool"""
        return self._run(tool_input)
    
class GitRepositoryPushTool(BaseTool):
    """Tool for pushing to a git repository"""
    name = 'git_repository_push'
    description = """
    Can be used to push to a git repository
    Input should be the local repository path
    Returns a string containing the result of the push
    """
    model: GitModel

    def _run(self, tool_input: str) -> str:
        """Run the tool"""
        return self.model.push(tool_input)

    async def _arun(self, tool_input: str) -> str:
        """Run the tool"""
        return self._run(tool_input)
    
class GitRepositoryAddFileTool(BaseTool):
    """Tool for adding a file to a git repository"""
    name = 'git_repository_add_file'
    description = """
    Can be used to stage a file to a git repository's index so it can be committed
    Input should be a comma separated list containing the local repository path followed the path to the file to add
    Returns a string containing the result of the add
    """
    model: GitModel

    def _run(self, tool_input: str) -> str:
        """Run the tool"""
        local_repository_path, file_path = tool_input.split(',', 1)
        # clean spaces
        local_repository_path = local_repository_path.strip()
        file_path = file_path.strip()
        return self.model.add_file(local_repository_path, file_path)

    async def _arun(self, tool_input: str) -> str:
        """Run the tool"""
        return self._run(tool_input)

class GitRepositoryCommitTool(BaseTool):
    """Tool for committing a git repository"""
    name = 'git_repository_commit'
    description = """
    Can be used to commit changes to a git repository
    Input should be a comma separated list containing the local repository path followed by the commit message
    Returns a string containing the result of the commit
    """
    model: GitModel

    def _run(self, tool_input: str) -> str:
        """Run the tool"""
        local_repository_path, message = tool_input.split(',', 1)
        # clean spaces
        local_repository_path = local_repository_path.strip()
        return self.model.commit(local_repository_path, message)

    async def _arun(self, tool_input: str) -> str:
        """Run the tool"""
        return self._run(tool_input)
    
class GitRepositoryGetFileContentTool(BaseTool):
    """Tool for getting the content of a file in a git repository"""
    name = 'git_repository_get_file_content'
    description = """
    Can be used to get the content of a file in a git repository
    Input should be a comma separated list containing the local repository path followed by the path to the file
    Returns a string containing the content of the file
    """
    model: GitModel

    def _run(self, tool_input: str) -> str:
        """Run the tool"""
        local_repository_path, file_path = tool_input.split(',')
        # clean spaces
        local_repository_path = local_repository_path.strip()
        file_path = file_path.strip()
        return self.model.get_file_content(local_repository_path, file_path)

    async def _arun(self, tool_input: str) -> str:
        """Run the tool"""
        return self._run(tool_input)

class GitRepositoryWriteFileTool(BaseTool):
    """Tool for writing a file to a git repository"""
    name = 'git_repository_write_file'
    description = """
    Can be used to write the content of a file to a git repository
    Input should be a comma separated list containing the local repository path followed by the path to the file, followed by the content of the file
    Returns a string containing the result of the write
    """
    model: GitModel

    def _run(self, tool_input: str) -> str:
        """Run the tool"""
        local_repository_path, file_path, content = tool_input.split(',', 2)
        # clean spaces
        local_repository_path = local_repository_path.strip()
        file_path = file_path.strip()
        return self.model.write_file(local_repository_path, file_path, content)

    async def _arun(self, tool_input: str) -> str:
        """Run the tool"""
        return self._run(tool_input)



    