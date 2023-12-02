from paperbox.cli.ollama import run_ollama_passthrough_pipe
from paperbox.io.markdown_management import (
    get_all_markdown_file_names,
    get_all_markdown_folder_names,
    add_markdown_folder,
    delete_markdown_folder,
    add_markdown_file,
    delete_markdown_file,
    rename_markdown_folder,
    rename_markdown_file,
    move_markdown_file,
    copy_markdown_file,
)
from paperbox.llm_pipelines.document_relevance_sorter import DocumentRelevanceSorter
from paperbox.io.markdown_document_utility import MarkdownDocumentUtility
from paperbox.llm_pipelines.ollama_markdown_rewriter import OllamaMarkdownRewriter
from paperbox.llm_pipelines.ollama_markdown_writer import OllamaMarkdownWriter
from paperbox.llm_pipelines.ollama_chat import OllamaChat
from langchain.schema.document import Document
from rich.console import Console
from rich.markdown import Markdown
from textwrap import dedent
from dataclasses import dataclass
import inquirer
import cmd


@dataclass
class CMDState(object):
    """A class to hold the state of the CLI."""

    markdown_utility: MarkdownDocumentUtility = None
    ollama_model_name: str = "mistral-openorca"


class Editor(cmd.Cmd):
    intro = dedent(
        """
            ____                        ____            
           / __ \____ _____  ___  _____/ __ )____  _  __
          / /_/ / __ `/ __ \/ _ \/ ___/ __  / __ \| |/_/
         / ____/ /_/ / /_/ /  __/ /  / /_/ / /_/ />  <  
        /_/    \__,_/ .___/\___/_/  /_____/\____/_/|_|  
                   /_/                                                                                
        """
    )
    prompt = "(paperbox) "
    boot_instructions = dedent(
        f"""
        \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n
        #######################################################\n
        Welcome to the Paperbox Editor!\n
        Type [bold magenta]help[/] to see the list of commands.\n
        #######################################################
        """
    )

    def onecmd(self, line: str) -> bool:
        """Override the onecmd method to catch exceptions."""
        try:
            return super().onecmd(line)
        except Exception as e:
            self.console.print(f"Error: {e}", style="bold red")
            return False  # Don't exit the CLI

    def preloop(self) -> None:
        self.console = Console()
        self.state = CMDState()
        self.console.print(self.boot_instructions, style="bold yellow")

    def do_a(self, line) -> None:
        """
        Add a section.
        Usage: a <section_description>
        """
        # --- Validate Loaded File ---
        if self.state.markdown_utility is None:
            self.console.print(
                "No file loaded. Use [bold magenta]load[/] to load a file.",
                style="bold yellow",
            )
            return
        # --- Add Until Satisfied ---
        md_writer = OllamaMarkdownWriter(
            ollama_model_name=self.state.ollama_model_name,
        )
        written_section_text = md_writer.write_section(instructions=line)
        self.console.print(
            Markdown(written_section_text),
            style="bold blue",
        )
        while True:
            satisfied = inquirer.prompt(
                [
                    inquirer.Confirm(
                        "satisfied",
                        message="Are you satisfied with the written section?",
                    )
                ]
            )["satisfied"]
            if satisfied:
                break
            instructions = inquirer.prompt(
                [
                    inquirer.Text(
                        "instructions",
                        message="Enter the instructions for the section.",
                    )
                ]
            )["instructions"]
            written_section_text = md_writer.write_section(instructions=instructions)
            self.console.print(
                Markdown(written_section_text),
                style="bold blue",
            )

        # --- Add the section to the document ---
        self.state.markdown_utility.loaded_documents.append(
            Document(
                page_content=written_section_text,
            )
        )
        # --- Save the document ---
        self.state.markdown_utility.save_to_disk()
        self.state.markdown_utility.load_from_disk()

    def do_d(self, line) -> None:
        """
        Delete a section.
        Usage: d <section_to_delete>
        """
        # --- Validate Loaded File ---
        if self.state.markdown_utility is None:
            self.console.print(
                "No file loaded. Use [bold magenta]load[/] to load a file.",
                style="bold yellow",
            )
            return
        if len(self.state.markdown_utility.loaded_documents) == 0:
            self.console.print(
                "No sections to delete. Use [bold magenta]a[/] to add a section.",
                style="bold yellow",
            )
            return
        # --- Get the section to delete ---
        doc_rel_sorter = DocumentRelevanceSorter(
            documents=self.state.markdown_utility.loaded_documents, top_k=3
        )
        relevant_sections = doc_rel_sorter.get_sorted_by_relevance_to_query(query=line)
        section_choices = [
            f"{self.state.markdown_utility.get_readable_header_from_document(section)}"
            for section in relevant_sections
        ]
        section_choices.append("Cancel")
        section_choice = inquirer.prompt(
            [
                inquirer.List(
                    "section",
                    message="Delete section identifier.",
                    choices=section_choices,
                )
            ]
        )["section"]
        if section_choice == "Cancel":
            return
        section_delete_index = self.state.markdown_utility.loaded_documents.index(
            relevant_sections[section_choices.index(section_choice)]
        )
        # --- Delete the section ---
        self.state.markdown_utility.loaded_documents.pop(section_delete_index)
        # --- Save the document ---
        self.state.markdown_utility.save_to_disk()
        self.state.markdown_utility.load_from_disk()

    def do_e(self, line) -> None:
        """
        Edit a file.
        Usage: e <section_to_edit>
        Follow Up: Enter the instructions for the section.
        """
        # --- Validate Loaded File ---
        if self.state.markdown_utility is None:
            self.console.print(
                "No file loaded. Use [bold magenta]load[/] to load a file.",
                style="bold yellow",
            )
            return
        if len(self.state.markdown_utility.loaded_documents) == 0:
            self.console.print(
                "No sections to edit. Use [bold magenta]a[/] to add a section.",
                style="bold yellow",
            )
            return
        # --- Get the section to edit ---
        doc_rel_sorter = DocumentRelevanceSorter(
            documents=self.state.markdown_utility.loaded_documents, top_k=3
        )
        relevant_sections = doc_rel_sorter.get_sorted_by_relevance_to_query(query=line)
        section_choices = [
            f"{self.state.markdown_utility.get_readable_header_from_document(section)}"
            for section in relevant_sections
        ]
        section_choices.append("Cancel")
        section_choice = inquirer.prompt(
            [
                inquirer.List(
                    "section",
                    message="Edit section identifier.",
                    choices=section_choices,
                )
            ]
        )["section"]
        if section_choice == "Cancel":
            return
        section_edit_index = self.state.markdown_utility.loaded_documents.index(
            relevant_sections[section_choices.index(section_choice)]
        )
        section_to_edit = self.state.markdown_utility.loaded_documents[
            section_edit_index
        ]

        # --- Edit Until Satisfied ---
        md_editor = OllamaMarkdownRewriter(
            section_to_rewrite=section_to_edit,
            ollama_model_name=self.state.ollama_model_name,
        )
        rewritten_section_text = ""
        while True:
            instructions = inquirer.prompt(
                [
                    inquirer.Text(
                        "instructions",
                        message="Enter the instructions for the section.",
                    )
                ]
            )["instructions"]
            rewritten_section_text = md_editor.rewrite_section(
                instructions=instructions
            )
            self.console.print(
                Markdown(rewritten_section_text),
                style="bold blue",
            )
            satisfied = inquirer.prompt(
                [
                    inquirer.Confirm(
                        "satisfied",
                        message="Are you satisfied with the rewritten section?",
                    )
                ]
            )["satisfied"]
            if satisfied:
                break
        # --- Replace the section in the document ---
        self.state.markdown_utility.loaded_documents[
            section_edit_index
        ].page_content = rewritten_section_text
        # --- Save the document ---
        self.state.markdown_utility.save_to_disk()
        self.state.markdown_utility.load_from_disk()

    def do_load(self, file_path: str) -> None:
        """Load a file to edit."""
        self.console.print(f"Loading file {file_path}", style="bold blue")
        self.state.markdown_utility = MarkdownDocumentUtility(file_path=file_path)

    def do_q(self, line) -> None:
        """Ask LLM a Question and Output the Answer."""
        # --- Ask LLM a Question ---
        ollama_chat = OllamaChat(ollama_model_name=self.state.ollama_model_name)
        ollama_chat.chat(chat=line)

    def do_ollama(self, line) -> None:
        """
        Ollama Passthrough.
        Run ollama with the given input and return the output.

        Help: ollama help
        """
        self.console.print(run_ollama_passthrough_pipe(line), style="bold blue")

    def do_get_ollama_model(self, _) -> None:
        """Get the current Ollama Model."""
        self.console.print(
            f"Current Ollama Model: {self.state.ollama_model_name}", style="bold blue"
        )

    def do_switch_ollama_model(self, line) -> None:
        """
        Switch the Ollama Model.
        Usage: switch_ollama_model <ollama_model_name>
        """
        self.state.ollama_model_name = line
        self.console.print(f"Switched to Ollama Model {line}", style="bold blue")

    def do_list_markdown_files(self, _) -> None:
        """List all markdown files."""
        self.console.print("\n".join(get_all_markdown_file_names()), style="bold blue")

    def do_list_markdown_folders(self, _) -> None:
        """List all markdown folders."""
        self.console.print(
            "\n".join(get_all_markdown_folder_names()), style="bold blue"
        )

    def do_add_markdown_folder(self, folder_name: str) -> None:
        """Add a markdown folder."""
        add_markdown_folder(folder_name)
        self.console.print(f"Added folder {folder_name}", style="bold blue")

    def do_delete_markdown_folder(self, folder_name: str) -> None:
        """Delete a markdown folder."""
        delete_markdown_folder(folder_name)
        self.console.print(f"Deleted folder {folder_name}", style="bold blue")

    def do_add_markdown_file(self, file_name: str) -> None:
        """Add a markdown file."""
        add_markdown_file(file_name)
        self.console.print(f"Added file {file_name}", style="bold blue")

    def do_delete_markdown_file(self, file_name: str) -> None:
        """Delete a markdown file."""
        delete_markdown_file(file_name)
        self.console.print(f"Deleted file {file_name}", style="bold blue")

    def do_rename_markdown_folder(self, old_new_folder: str) -> None:
        """Rename a markdown folder."""
        rename_markdown_folder(*old_new_folder.split(" "))
        self.console.print(
            f"Renamed folder {old_new_folder.split(' ')[0]} to {old_new_folder.split(' ')[1]}",
        )

    def do_rename_markdown_file(self, old_new_file: str) -> None:
        """Rename a markdown file."""
        rename_markdown_file(*old_new_file.split(" "))
        self.console.print(
            f"Renamed file {old_new_file.split(' ')[0]} to {old_new_file.split(' ')[1]}",
        )

    def do_move_markdown_file(self, old_new_file: str) -> None:
        """Move a markdown file."""
        move_markdown_file(*old_new_file.split(" "))
        self.console.print(
            f"Moved file {old_new_file.split(' ')[0]} to {old_new_file.split(' ')[1]}",
        )

    def do_copy_markdown_file(self, old_new_file: str) -> None:
        """Copy a markdown file."""
        copy_markdown_file(*old_new_file.split(" "))
        self.console.print(
            f"Copied file {old_new_file.split(' ')[0]} to {old_new_file.split(' ')[1]}",
        )

    def do_exit(self, _) -> bool:
        """Exit the CLI."""
        self.console.print("[italic red]Exiting PaperBox...[/]", style="italic blue")
        return True  # Exits the CLI

    def default(self, line) -> None:
        self.console.print(
            f"Command [bold red]{line}[/] not recognized", style="bold yellow"
        )
