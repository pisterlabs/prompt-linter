"""Python Wrapper-like application interface for autogpt for Obsidian.md"""

"""perate using operating system of the docker container"""
import os

"""the library `python-frontmatter` for working with frontmatters of files"""
import frontmatter

"""For Type Checking and Lists"""
from typing import TYPE_CHECKING, List

"""Import the guidance llm python library for LLM prompting"""
import guidance

"""Operating System operations allowing for operating system integation"""
import os

"""Github Operations allowing for control of Git Operations for the Obsidian Vault"""
from git.repo import Repo

"""Operating System Operations for Working with Files and Directories within the Operating System"""
import os

"""Guidance Library for Prompt LLM Models with a more Predictable, Definable Output"""
import guidance

"""Import the datetime Module for Accurate Time"""
import datetime


class obsidianNote():
    """Note object allowing for inteactions with markdown notes with the syntax of Obsidian for the AutoGPT plugin, AutoGPTObsidian"""

    def __init__(self, vault: obsidianVault, note_path) -> None: 
        """
        Initialize the Obsidian_Note object with the given vault_path inside of the AutoGPT Workspace.

        Args:
            path: The path of the note with respect to the vault_working_directory. (Not including the vault_directory)
        """
        # Vault : The `Obsidian_Vault` object assigned with the Note Object.
        self.vault = vault # Vault provides access to the notes in the vault.
        # Note Title : The Title of the file/note defined by the path of the note/file. 
        self.path = os.path.join(vault.vault_directory, note_path)
        # Note Title : the Title of the Obsidian_Note
        self.title = os.path.basename(self.path).replace(".md", "")  

        with open(os.path.join(vault.vault_directory, note_path)) as f:
            self.frontmatter = frontmatter.load(f)
            self.update_note_attributes()
        

    def reload(self) -> None: 
        """Reloads the note content from the file at the given path"""
        with open(self.path, "r") as f: 
            self.content = f.read()
        if self.content is None: 
            self.content = ""
        self.update_note_attributes()

    def update_note_attributes(self) -> None: 
        """
        Updates the attributes of the Obsidian_Note object based on the content of the note give as context, a string.
        Updates the following attributes: tags, incoming_links, outgoing_links, frontmatter, dataview_metadata, and dataview_metadata_fields 

        Parameters: 
            context : string representing the the content of the note to update the ntoe attributes on 
        """
        with open(self.path, "r") as file: 
            self.content = file.read()
        self.tags = self.get_tags()
        


    def get_tags(self) -> list:
        """
        Retreives the tags of the note defined in Frontmatter syntax, obsidian content syntax, and dataview syntax. It first checks the frontmatter, then the content, then for tags defined in the dataview syntax.

        Returns: 
            list : A list of tags marked with the note for this Obsidian_Vault Object
        """
        self.update_note_attributes()
        ### FRONTMATTER ###
        tags = []
        ### CONTENT ###
        for line in self.content.split("\n"): 
            # split around spaces
            tags.extend(
                word.replace("#", "")
                for word in line.split(" ")
                if word.startswith("#")
            )
        ### DATAVIEW ### 
        for line in self.content.split("\n"): 
            if line.startswith("tags::"): 
                tags += line.replace("tags::", "").strip().split(" ")
        return tags


class Obsidian:
    def __init__(self):
        """Initializes an Obsidian object with a vault inside a Debian Docker Container."""
        self.vault = obsidianVault()
        self.vault_path = os.path.join(
            os.path.expanduser("~"),
            f"autogpt{os.sep}auto_gpt_workspace{os.sep}{self.vault.vault_name}",
        )

    def _find_note_by_title(self, title: str) -> obsidianNote | None:
        return self.vault.find_by_title(title)

    def _create_note(self, title: str, content: str) -> Exception | None:
        """
        Create a note inside the vault with a title, content, tags, type,  and a summary within the frontmatter of said note following the Obsidian format
        Parameters:
            - title: The title of the note. In other words, the title is the name of the note. The title is also the name of the file to be produced.
            - content: the content of the note to be written in Markdown. In other words, the content of a note is the text to be placed in the main body of the note excluding the frontmatter.
        Returns:
            - the created note content.

        """
        # Sync the vault to most recent changes
        self.vault.sync_vault()

        guidance.llm = guidance.llms.OpenAI("text-davinci-003")

        # geneal path ~/
        if self.vault.git_url is None:
            return Exception(
                "Vault is not a git repository. Please make sure that the vault is a git repository."
            )

        self.vault.git_url.split("/")[-1]

        note = self._find_note_by_title(title)
        if note is not None and note.title == title:
            return Exception(f"Note with title {title} exists in the vault.")

        note = self._find_note_by_title(title)

        if title.endswith(".md"):
            title = title[:-3]
            title = f"{title}.md"

        note = open(os.path.join(self.vault.vault_path, title), "w")
        note = open(os.path.join(self.vault.vault_path, title), "w")

        note = open(os.path.join(self.vault.vault_path, title), "w")

        current_date = datetime.datetime.now()
        valid_tags_file = open(
            os.path.join(self.vault.vault_path, "valid_tags.txt"), "w"
        )
        create_note_program = guidance(
            """
            {{#system~}}
            You are a writer. You are writing a note. The note is about {title}. The note is about {gen 'type'}.
            The note has the following {{content}} a parameter to this function. The current_date is {{current_date}}.
            The note is in the vault {{vault_path}}.
            You must create a summary, classify the note by type, and choose from valid tags found to be within the vault already.
            Place it in the frontmatter in correct Markdown format to follow.
            {{~/system~}}
            ---
            title: {{title}}
            tags: {{#select 'tags' options = valid_tags}}
            summary: {{gen 'summary'}}
            created: {{current_date}}
            type: {{gen 'type'}}
            ---
            {{content}}
        """
        )
        executed_create_program = create_note_program(
            content=content,
            valid_tags=self.vault.get_valid_tags(),
            current_date=current_date,
            vault_path=self.vault.vault_path,
            title=title,
        )

    def _create_markdown_file(self, note_path: str, content: str) -> Exception | None:
        """
        Creats a markdown file and Obsidian Note housed inside the obsidian vault and the Obsidian_Vault object respectively.

        Parrameters:
        - note_path : name of the file to crerate and the title of the note to create within the obsidian vault

        """
        if not note_path.endswith(".md"):
            note_path += ".md"
        try:
            self._sync_vault()
            note = obsidianNote(self.vault, note_path)
        except Exception as e:
            return e

        return None

    def _sync_vault(self) -> str:
        """
        Syncs the vault with the remote repository.

        Returns:
            - string representing the actions/operation success or failures.
        """
        try:
            self.vault.sync_vault()
            return "Vault synced successfully."
        except Exception as e:
            return f"Vault failed to sync because of the following error: {str(e)}"

    def _create_note_flashcards(self, title: str) -> Exception | None:
        """
        Create Spaced-Repetition Styled Flashcards inside the vault with title,
        content, tags, and summary. The title of the note from which flashcards
        are being generated + "Flashcards" + number of flashcards for that note
        in the vault that were found to be in the vault to avoid name collisions.

        Parameters:
            - title : the title of the note to create inside of the obsidian vault
        """

        self._sync_vault()

        if title is None:
            return Exception("Title cannot be None.")
        note = self._find_note_by_title(title)
        if note is not None and note.content is not None:
            return Exception("Note with title " + title + " exists in the vault.")
        executed_flashcards_program = guidance(
            """
                A conversation with an AI for use in Obsidian.
                You are an AI that is helping write flashcards for the purpose of spaced repetition.
                Flashcards should have the following format:
                '<question>
                ?
                <answer>'.
                Here is an example flashcard:
                ${exampleFlashcard}
                Do NOT number the flashcards. Make sure each question you state is atomic, so that it only asks one thing, i.e. avoids the use of 'and'. You do not need to write flashcards on everything in the document, just start with the most important. You will be a writing it on the subject matter of:'${title}.
            """
        )
        self._create_note(title, executed_flashcards_program)

class obsidianVault: 
    """
    Vault Object allowing for interactions with markdown notes specifically with the syntax 
    of Obsidian Markdown for the AutoGPT plugin, AutoGPT-Obsidian.

    This metaphorical obsidian vault class has a few native components on itself: 
            `markdown content` : The Markdown Files in the vault.  
            `vault path`       : The path of the vault within the file system.
            `vault name`       : The name of the vault. 
            `git url`          : The git url of the vault. 
            `git api key`      : The git api key of the vault. 
            `git username`     : The git username of the vault.
    """
    def __init__(self) -> None:
        """Initializes the Obsidian Vault Object"""
        # Retrieve the environment variables for the obsidian plugin
        self.initialize_environment_variables()
        self.sync_vault()
        self.vault_directory = os.sep.join([ os.path.expanduser("~"), "autogpt", "auto_gpt_workspace", self.vault_name ])
        self.content = self.get_vault_content()

        
    def initialize_environment_variables(self) -> Exception|None:
        """Initializes the Environment Variables for the Obsidian Plugin"""
        if os.getenv("OBSIDIAN_GITHUB_API_KEY"):
            self.git_api_key = os.getenv("OBSIDIAN_GITHUB_API_KEY")
        else: 
            return Exception("Please set the OBSIDIAN_GITHUB_API_KEY environment variable in the .env file.")

        if os.getenv("OBSIDIAN_VAULT_GIT_URL"):
            self.git_url = os.getenv("OBSIDIAN_VAULT_GIT_URL") 
        else: 
            return Exception("Please set the OBSIDIAN_VAULT_GIT_URL environment variable in the .env file.")

        if os.getenv("OBSIDIAN_GITHUB_USERNAME"):
            self.git_username = os.getenv("OBSIDIAN_GIT_USERNAME")
        else: 
            return Exception("Please set the OBSIDIAN_GIT_USERNAME environment variable in the .env file.")

        if os.getenv("OBSIDIAN_VAULT_NAME"): 
            self.vault_name = str(os.getenv("OBSIDIAN_VAULT_NAME"))
        else:
            return Exception("Please set the OBSIDIAN_VAULT_NAME environment variable in the .env file.")

        if os.getenv("OBSIDIAN_VAULT_PATH"): 
            self.vault_path = str(os.getenv("OBSIDIAN_VAULT_PATH")) 
        else:
            return Exception("Please set the OBSIDIAN_VAULT_PATH environment variable in the .env file.")

        if os.getenv("OBSIDIAN_GITHUB_API_KEY"): 
            self.git_api_key = os.getenv("OBSIDIAN_GITHUB_API_KEY")
        else: 
            return Exception("Please set the OBSIDIAN_GITHUB_API_KEY environment variable in the .env file.")

        if os.getenv("OBSIDIAN_FLASHCARD_DIRECTORY"): 
            self.flashcard_directory = os.getenv("OBSIDIAN_FLASHCARD_DIRECTORY")
            return None
        else: 
            return Exception("Please set the OBSIDIAN_FLASHCARD_DIRECTORY environment variable in the .env file.")

    def clone_vault(self) -> Exception|None:
        """Clones the Vault from the Vaults Git Url into the Workspace"""
        # Create the working directory
        working_directory = os.path.join(os.path.expanduser("~"), "autogpt"+ os.sep  +  "auto_gpt_workspace" + os.sep +  self.vault_name)
        os.makedirs(working_directory, exist_ok=True)
        try: 
            Repo.clone_from(self.git_url, working_directory)
            return None
        except Exception as e:
            return e

    def search_vault_title(self, title: str) -> obsidianNote|None:
        """ 
        Searches the Obsidian Vault's content for an Obsidian_Note with a given title

        Parameters: 
                title: the Title to `Search` for inside the Vault.
        """
        # For each file with a `.md` extension as target:
        # for ob_note in self.markdown_content: 
        # List result = 
        for note in self.content:
            if note.title == title: 
                return note

        return None

    def sync_vault(self) -> Exception|None: 
        """ 
        Sync the Obsidian Vault within the workspace with the remote Git repository.
        If there is a vault in the workspace, then it syncs the vault with the remote 
        Git repository. If not, then it clones the remote Git repository into the workspace.

        Returns: 
            Exception or None for an exception thrown or successful operation
        """
        git_url = os.getenv("OBSIDIAN_VAULT_GIT_URL") 
        git_api_key = os.getenv("OBSIDIAN-GITHUB_API_KEY")
        git_username = os.getenv("OBSIDIAN-GITHUB_USERNAME")
        if self.git_url is None:
            return Exception("2319 - No Git URL provided.") 
        split_url = self.git_url.split("//")
        auth_repo_url = f"//{git_username}:{git_api_key}@".join(split_url)

        repo = Repo(self.vault_directory)
        origin = repo.create_remote("origin", repo.remotes.origin.url)
        assert origin.exists()
        assert origin == repo.remotes.origin == repo.remotes["origin"]
        # There are changes in the remote repository, then pull the latest changes from the remote repository.
        if repo.head.commit != repo.remotes.origin.fetch()[0].commit: 
            repo.git.pull()
            try:
                repo.git.add(update=True) 
                repo.git.commit("-m", "Auto-commit from AutoGPT") 
                origin = repo.remote(name="origin")
                origin.push() 
                return None 
            except Exception as e: 
                return e

        # If there are changes in repo
        elif repo.is_dirty():
            repo.git.add(update=True) 
            repo.git.commit("-m", "Auto-commit from AutoGPT") 
            origin = repo.remote(name="origin")
            try:
                origin.push()
                print(f"""Pushed changes to {git_url}""")
                return None 
                
            except Exception as e: 
                return e

    def get_valid_tags(self) -> list: 
        """ 
        Retrieves a list of valid tags from the vault. 

        Returns: a list of valid tags used within the vault.
        """ 
        # Set guidance's OpenAI Model to "text-davinci-004"
        guidance.llm = guidance.llms.OpenAI("gpt4")

        # Sync the vault to most recent changes
        self.sync_vault()

        # Create a list of valid tags from the vault
        valid_tags = []

        # Iterate through the vault to find all tags
        for root, [], files in os.walk(self.vault_path):
            for file in files:
                if file.endswith(".md"):
                    with open(os.path.join(root, file), "r") as f:
                        for line in f.readlines():
                            if line.startswith("tags:"):
                                for tag in line.split(":")[0].split(","):
                                    valid_tags.append(tag.strip())

        # Additionally, Obsidian allows for tags to be defined as #tags, so we will add those to the list of valid tags.
        for root, [], files in os.walk(self.vault_path):
            for file in files:
                if file.endswith(".md"):
                    with open(os.path.join(root, file), "r") as f:
                        for line in f.readlines():
                            for word in line.split(" "):
                                if word.startswith("#"):
                                    valid_tags.append(word.strip())
        return valid_tags

# A function to get the content of the vault returning a list Obsidian_Note objects 
    def get_vault_content(self) -> list:
            """A Basic Method to get the Content of the Vault returning a list Obsidian_Note Objects"""
            notes = []

            for root , [] , files in os.walk(self.vault_path):
                for file in files:
                    if file.endswith(".md"):
                        with open(os.path.join(root, file), "r"):
                            # Create a new Obsidian_Note object with a reference to the Obsidian_Vault object
                            note = obsidianNote(self, note_path=os.path.join(root, file))
                            notes.append(note)
            return notes

    def smart_seach(self, topic: str):
        pass

    def find_by_title(self, title:str) -> obsidianNote|None:
        """Finds an Obsidian_Note by its Title"""
        notez = None
        for note in self.content:
            if (note.title == title):
                notez = note
                return notez
        return None  
    def create_note(self, note_path: str, content: str = "") -> obsidianNote:
        """Creates a new Obsidian_Note in the Vault"""
        note = obsidianNote(self, note_path )
        return note

