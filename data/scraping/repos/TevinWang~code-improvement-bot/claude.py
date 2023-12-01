import os, subprocess, threading
from tqdm import tqdm

from anthropic import AI_PROMPT, HUMAN_PROMPT, Anthropic
from dotenv import load_dotenv

from github_scraper import fetch_python_files_from_github_url

load_dotenv()
CLAUDE_API = os.getenv("CLAUDE_API")

def main(repo_url):
    # scrape
    github_url = repo_url
    python_files = fetch_python_files_from_github_url(github_url)
    python_files2 = python_files.copy()
    # ASSUMING OUTPUT IS A LIST OF TUPLES OF (FILE_PATH, [FILE LINES])
    for i, file in enumerate(tqdm(python_files)):
        context_list = []
        threads = []
        to_react = []

        add = 1
        to_react_line = []
        to_react_number = []
        for j, line in enumerate(file[1].splitlines()):
            if len(line.strip()) < 10:
                continue
            if (not line):
                continue
            if (line[0].strip() == "#"):
                continue
            if add % 3 != 0:
                line = list(line)
                for k, char in enumerate(line):
                    if char == "\"":
                        line[k] = "\'"
                line = "".join(line).strip()
                to_react_number.append(str(j))
                to_react_line.append(line)
            else:
                line = list(line)
                for k, char in enumerate(line):
                    if char == "\"":
                        line[k] = "\'"
                line = "".join(line).strip()
                to_react_number.append(str(j))
                to_react_line.append(line)
                to_react.append(", ".join(to_react_number) + "|||" + ", \n".join(to_react_line))

                to_react_number = []
                to_react_line = []

            add += 1
        # def start(to_react, cur_i):
        #     p = subprocess.Popen(['node', 'add_doc_context.js', "|||||".join(to_react)], stdout=subprocess.PIPE)
        #     out = p.stdout.read()
        #     # expected: line1|||line|||context|||||..
        #     context_list = [tuple(line.split("|||")) for line in out.decode("utf-8").split("|||||")]
        #     python_files[i] += (context_list,)

        # thread = threading.Thread(target=start, args=(to_react, i))
        # threads.append(thread)
        # thread.start()

    # print(threads)
    # for thread in threads:
    #     thread.join()

    # # python_files is now [(FILE_PATH, [FILE LINES], [(LINE NUMBER, LINE, CONTEXT)])]
    with open("python_files2.txt", "w", encoding="utf-8") as f:
        f.write('<files>')
        f.writelines(
            f"<file>\n<file_path>{file_path}</file_path>\n<file_content><![CDATA[\n{file_content}\n]]></file_content>\n</file>\n"
            for file_path, file_content in python_files2
        )
        f.write('</files>')
    # write the file
    # with open("python_files.txt", "w", encoding="utf-8") as f:
    #     f.write('<files>\n')
    #     for file_path, file_content, file_context in python_files:
    #         f.write(f"<file>\n<file_path>{file_path}</file_path>\n<file_content>\n{file_content}\n</file_content>\n<file_context>\n")
    #         for context in file_context:
    #             f.write(f"<line>\n<line_number>{context[0]}</line_number>\n<line_content>{context[1]}</line_content>\n<context>\n{context[2]}</context>\n</line>\n")
    #         f.write("</file_context>\n</file>\n")
    #     f.write('</files>')

    # # LOOKS LIKE
    # """
    # <files>
    #     <file>
    #         <file_path>path/to/file.py</file_path>
    #         <file_content>
    #             import os, subprocess
    #             etc
    #             etc
    #         </file_content>
    #         <file_context>
    #             <line>
    #                 <line_number>1</line_number>
    #                 <line_content>import os, subprocess</line_content>
    #                 <context>
    #                     import os, subprocess
    #                     context here
    #                 </context>
    #             </line>
    #             <line>
    #                 <line_number>2</line_number>
    #                 etc
    #                 etc
    #             </line>
    #         </file_context>
    #     </file>
    #     <file>
    #         etc
    #         etc
    #     </file>
    # </files>
    # """

    # read the file
    with open("python_files.txt", "r", encoding="utf-8") as f:
        python_files = f.read()


    prompt = f"""{HUMAN_PROMPT}

    Description:
    In this prompt, you are given a open source codebase that requires thorough cleanup, additional comments, and the implementation of documentation tests (doc tests). Your task is to enhance the readability, maintainability, and understanding of the codebase through comments and clear documentation. Additionally, you will implement doc tests to ensure the accuracy of the documentation while also verifying the code's functionality.

    Tasks:

    Codebase Cleanup:

    Identify and remove any redundant or unused code.
    Refactor any convoluted or confusing sections to improve clarity.
    Comments and Documentation:

    Add inline comments to explain complex algorithms, logic, or code blocks.
    Document the purpose, input, output, and usage of functions and methods.
    Describe the role of key variables and data structures used in the code.
    Doc Tests Implementation:

    Identify critical functions or methods that require doc tests.
    Write doc tests that demonstrate the expected behavior and output of the functions.
    Ensure the doc tests cover various scenarios and edge cases.
    Function and Variable Naming:

    Review function and variable names for clarity and consistency.
    Rename functions and variables if needed to improve readability.
    Readme File Update (Optional):

    Update the README file with a summary of the codebase and its purpose.
    Provide clear instructions for running the code and any dependencies required.
    Note:

    The codebase provided may lack sufficient comments and documentation.
    Focus on making the code easier to understand for others who read it in the future.
    Prioritize clarity and conciseness when writing comments and documentation.
    Implement doc tests using appropriate testing frameworks or methods.
    Ensure that the doc tests cover various scenarios to validate the code's correctness.
    This prompt allows the LLM to work on improving codebase quality through comments and documentation while also implementing doc tests for verification. Cleaning up and enhancing codebases in this way is a crucial skill for any developer, as it facilitates teamwork, code maintenance, and future development efforts.Claude, I'm seeking your expertise in adding comments and doc tests to Python code files.:

    Provide the updated code in a xml structure where your entire response is parseable by xml:

    <root>
    <diff>
    <!--Ensure the diff follows the unified diff format that would be returned by python difflib, providing clear context and line-by-line changes for ALL files.
    Give line numbers with the first line of the file content being line 1,
    ONLY CHANGE LINES OF FILE CONTENT (NOT ANY OF THE XML TAGS). Do this for all files.
    Add the entire thing as a cdata section '<![CDATA['
    This is what it is supposed to look like per file:
    --- a/path/to/file.txt (make sure to include the 'a/' in the path, and exactly 3 +s)
    +++ b/path/to/file.txt (make sure to include the 'b/' in the path, and exactly 3 -s)
    @@ -1,4 +1,4 @@ (ANYTHING after the @@ MUST BE ON A NEW LINE)
    This is the original content.s
    -Some lines have been removed.
    +Some lines have been added.
    More content here.
    Remove this comment and add the diff patch contents in the diff tag directly. DO NOT ADD THIS IN THE COMMENT
    -->

    </diff>
    [NO MORE DIFF SYNTAX AFTER THE DIFF TAG]
    <title>
    <!-- Relevant emoji + Include a github pull request title for your changes -->
    </title>
    <changes>
    <!-- Include details of the changes made in github BULLET POINTS, not xml, with some relevant emojis -->
    </changes>
    </root>

    Your focus should be on pythonic principles, clean coding practices, grammar, efficiency, and optimization. Do not change the file if you don't know what to do.

    Before you make a change, evaluate the following:
    - The code must work and stay valid
    - The code doesn't add any duplicate code that isn't necessary
    - The code has the right indentation for python
    - The code works
    If one of these is not valid, do not add the change.

    Reminder to add the entire diff as a cdata section '<![CDATA[' (not individually)

    Make sure to add ANYTHING after the @@ ON A NEW LINE
    Be sure to add ANYTHING after the @@ ON A NEW LINE

    Be sure to add changes to all files provided.

    Reminder that the title should contain a relevant emoji and be github style. The changes section should include changes in bullet points.

    Please find the files for review and modification below. They also contain the relevant context and documentation from python to help guide you.
    {python_files}

    Remember the output is in the form: <root>
    <diff>
    </diff>
    <title>
    </title>
    <changes>
    </changes>
    </root>

    DO NOT STOP IN THE MIDDLE.
    Now act as a XML code outputter. Generate based off of the entire instructions, do not cut out in the middle (remember to populate the patch in the diff section). Do not add any additional context or introduction in your response, make sure your entire response is parseable by xml.
    {AI_PROMPT}"""

    anthropic = Anthropic(
        api_key=CLAUDE_API,
    )
    completion = anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=10000,
        prompt=prompt,
    )

    with open("completion_output.xml", "w", encoding="utf-8") as file:
        file.write(completion.completion)

if __name__ == "__main__":
    main("https://github.com/tevinwang/lancedb")