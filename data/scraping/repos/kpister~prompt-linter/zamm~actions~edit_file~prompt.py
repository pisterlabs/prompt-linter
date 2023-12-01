from langchain.prompts.prompt import PromptTemplate

WHICH_FILE_PROMPT = PromptTemplate(
    input_variables=[],
    template="You decide to edit the file located at: `",
)

NEW_CONTENTS_PROMPT = PromptTemplate(
    input_variables=["file_path"],
    template="""
You decide to edit the file `{file_path}`. It currently does not exist.

You write this content out to the file:

```
""".lstrip(),
)

REPLACE_CONTENTS_PROMPT = PromptTemplate(
    input_variables=["file_path", "old_contents"],
    template="""
You decide to edit the file `{file_path}`. Its current contents are

```
{old_contents}
```

You replace the file contents with

```
""".lstrip(),
)

NEW_FILE_LOGGER = PromptTemplate(
    input_variables=["file_path", "new_contents"],
    template="""
You decide to edit the file `{file_path}`. It doesn't yet exist.

You write out to the file the contents

```
{new_contents}
```
""".rstrip(),
)

EDIT_LOGGER = PromptTemplate(
    input_variables=["file_path", "old_contents", "new_contents"],
    template="""
You decide to edit the file `{file_path}`. Its old contents were

```
{old_contents}
```

You replace the file contents with

```
{new_contents}
```
""".rstrip(),
)

CONDENSED_LOGGER = PromptTemplate(
    input_variables=["file_path"],
    template="""
You have edited `{file_path}` as per instructions.
""".rstrip(),
)
