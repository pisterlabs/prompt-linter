import pathlib
import subprocess
import tempfile

from langchain.docstore.document import Document


def get_github_docs(repo_owner, repo_name):
    with tempfile.TemporaryDirectory() as d:
        subprocess.check_call(
            f'git clone --depth 1 https://github.com/{repo_owner}/{repo_name}.git .',
            cwd=d,
            shell=True,
        )
        git_sha = (
            subprocess.check_output('git rev-parse HEAD', shell=True, cwd=d)
            .decode('utf-8')
            .strip()
        )
        repo_path = pathlib.Path(d)
        markdown_files = list(repo_path.glob('*/*.md')) + list(
            repo_path.glob('*/*.mdx'),
        )
        for markdown_file in markdown_files:
            with open(markdown_file) as f:
                relative_path = markdown_file.relative_to(repo_path)
                github_url = f'https://github.com/{repo_owner}/{repo_name}/blob/{git_sha}/{relative_path}'
                yield Document(page_content=f.read(), metadata={'source': github_url})
