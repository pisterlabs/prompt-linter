from langchain.schema import BaseOutputParser

class FormattedOutputConvertToText(BaseOutputParser):
    """
    Parses and formats the output of a language model as a simple string.
    """

    def parse(self, text: str):
        """
        Formats the input text as a simple string.

        Parameters:
        - text (str): The text output from a language model call.

        Returns:
        str: The formatted text.
        """
        
        # Directly convert the text to string without additional formatting
        formatted_text = f"{text}"
        return formatted_text
    

class MarkdownTreeStructureOutputParser(BaseOutputParser):
    """
    Parses and formats the output of a language model to represent a repository structure in Markdown.
    """

    def parse(self, text: str):
        """
        Formats the repository structure as a tree in Markdown.

        Parameters:
        - text (str): The text output from a language model call, representing file paths.

        Returns:
        str: The formatted tree structure in Markdown.
        """
        # Split the text into lines, each representing a file path
        paths = text.split("\n")

        # Organize paths into a tree structure
        tree = self.build_tree_structure(paths)

        # Format the tree structure for Markdown output
        formatted_text = "```\n" + self.format_tree(tree) + "```"
        return formatted_text

    def build_tree_structure(self, paths):
        """
        Builds a tree structure from a list of file paths.

        Parameters:
        - paths (list): A list of file paths.

        Returns:
        dict: A nested dictionary representing the tree structure of the paths.
        """
        tree = {}
        for path in paths:
            current_level = tree
            for part in path.split('/'):
                if part not in current_level:
                    current_level[part] = {}
                current_level = current_level[part]
        return tree

    def format_tree(self, tree, indent=0):
        """
        Recursively formats the tree structure for Markdown output.

        Parameters:
        - tree (dict): The tree structure to format.
        - indent (int): The current indentation level for formatting.

        Returns:
        str: The formatted tree structure as a string.
        """
        output = ""
        for key, value in tree.items():
            output += "    " * indent + f"{key}\n"
            if isinstance(value, dict):
                output += self.format_tree(value, indent + 1)
        return output


class FormattedOutputParserSummary(BaseOutputParser):
    """
    Parses and formats the output of a language model to highlight file paths and their summaries.
    """

    def parse(self, text: str):
        """
        Formats the input text to highlight file paths and summaries.

        Parameters:
        - text (str): The text output from a language model call, containing file paths and summaries.

        Returns:
        str: The formatted text with file paths in bold and summaries as paragraphs.
        """
        # Assuming the text format is "File Path: {file_path} Summary: {summary}"
        parts = text.split("\nSummary: ")
        file_path = parts[0].replace("File Path: ", "").strip()
        summary = parts[1].strip() if len(parts) > 1 else ""

        # Format the file path in bold and the summary as a paragraph
        formatted_text = f"**{file_path}** \n\n{summary}"
        return formatted_text
