from langchain.text_splitter import RecursiveCharacterTextSplitter


class LkmlTextSplitter(RecursiveCharacterTextSplitter):
    """Attempts to split the text along LookML syntax."""

    def __init__(self, **kwargs):
        separators = [
            # First, try to split along class definitions
            "view: ",
            "explore: ",
            "\n  join: ",
            "\n  derived_table: ",
            "\n    sql: ",
            "\n  dimension: ",
            "\n  measure: ",
            "\n  set: ",
            "\n\n",
            "\n",
            " ",
            "",
        ]
        super().__init__(separators=separators, **kwargs)
