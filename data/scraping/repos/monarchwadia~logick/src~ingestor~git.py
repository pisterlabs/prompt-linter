from urllib.parse import urlparse
from git import Repo
from langchain.document_loaders import GitLoader
import os
from tempfile import TemporaryDirectory, mkdtemp
import logging as log

class GitIngestor():
    def from_disk(self, folder_path: str):
        log.debug("Loading repo from disk path:", folder_path)

        # Workaround for issue where files in git repos in .cache are not loaded because
        # git ignore-check uses this project's root /.gitignore instead of using the .gitignore of the target repo
        # so we copy the files to a temp dir and load from there
        temp_dir = mkdtemp()
        # with TemporaryDirectory() as temp_dir:
            # copy files from folder_path to dir
        log.debug("Copying files to temp dir:", temp_dir)
        os.system(f"rsync -rtv {folder_path}/ {temp_dir}/")
        branch = Repo(temp_dir).active_branch.name
        log.debug("Active branch:", branch)
        return GitLoader(repo_path=temp_dir, branch=branch)

    def from_web(self, url: str):
        cache_dir = self.__build_cache_dirname(url)
        # if cache directory does not exist, create it

        if os.path.exists(cache_dir):
            log.debug("Cache directory exists:", cache_dir)
        else:
            log.debug("No cache found. Loading repo from web:", url)
            os.makedirs(cache_dir, exist_ok=True)
            log.debug("Created cache directory:", cache_dir)
            log.debug("Cloning repo from:", url)
            Repo.clone_from(url, to_path=cache_dir)
            log.debug("Successfully cloned to:", cache_dir)

        return self.from_disk(cache_dir)

    
    def __build_cache_dirname(self, url: str):
        parsed_url = urlparse(url)
        namespace = parsed_url.path.strip('/').split('/')[-2:]
        cache_dir = os.path.join('./.cache', *namespace)
        return cache_dir