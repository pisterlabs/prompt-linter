import os
import argparse
import subprocess
from textwrap import dedent

from typing import List, Optional
import guidance
from typing import List, Dict, Tuple
import contextlib
import os
from pathlib import Path
from textwrap import dedent
import guidance

gpt4 = guidance.llms.OpenAI("gpt-4")
gpt35turbo = guidance.llms.OpenAI("gpt-3.5-turbo-16k")

rerun = False

def run_llm_program(program, *args, **kwargs):
    with open("log.txt", "a") as f, contextlib.redirect_stdout(
        f
    ), contextlib.redirect_stderr(f):
        return program(*args, **kwargs)


def build_paragraph(content):
    score = guidance(dedent("""
    {{#system~}}
    You summarize markdown blog posts.
    {{~/system}}
    {{#user~}}
   

    Post:
    ---
    {{content}} 
    ---
    Can you summarize this blog post in a three word sentence of the form 'This article is about ....'? Do no put the summary in quotes.
    Examples:
    - This article is about gmail API changes.
    - This article is about Python CLIs. 
    - This article is about Python CLIs. 
                            
    Can you summarize this blog post in a three word sentence of the form 'This article is about ....'? 
    {{~/user}}
    {{#assistant~}}
    {{gen 'summary' max_tokens=100 temperature=0}}
    {{~/assistant}}
    """),llm=gpt4)
    out = run_llm_program(score, content=content)
    article_sentence = out["summary"].strip()

    score = guidance(dedent("""
    {{#system~}}
    You summarize markdown blog posts.
    {{~/system}}
    {{#user~}}
   

    Post:
    ---
    {{content}} 
    ---
    Can you provide a short sentence explaining why Earthly would be interested to readers of this article? Earthly is an open source build tool for CI. The sentence should be of the form 'Earthly is popular with users of bash.' 
    {{~/user}}
    {{#assistant~}}
    {{gen 'summary' max_tokens=100 temperature=0}}
    {{~/assistant}}
    """),llm=gpt4)
    out = run_llm_program(score, content=content)
    tie_in_sentence = out["summary"].strip().split(".",1)[0]

    template = dedent(f"""
        **We're [Earthly](https://earthly.dev/). We make building software simpler and therefore faster using containerization. {article_sentence} {tie_in_sentence}. [Check us out](/).**
                """).strip()
    return template

def add_paragraph_if_word_missing(filename, dryrun):
    # Read the file
    with open(filename, 'r') as file:
        content = file.read()

    # Identify the frontmatter by finding the end index of the second '---'
    frontmatter_end = find_nth(content, '---', 2)

    # If frontmatter end exists
    if frontmatter_end != -1:
        frontmatter = content[:frontmatter_end + len('---')]  # Include '---' in frontmatter
        rest_of_file = content[frontmatter_end + len('---'):]  # rest_of_file starts after '---'
    else:
        frontmatter = ''
        rest_of_file = content

    if "funnel:" in frontmatter or "News" in frontmatter or " Write Outline" in rest_of_file or "topcta: false" in frontmatter:
        print(f"{filename}:Is Earthly focused, skipping.")
        return
    elif "iframe" in rest_of_file:
        print(f"{filename}:Youtube CTA, skipping.")
        return
    else:
        first_paragraph_found = False
        paragraphs = rest_of_file.split("\n")
        for paragraph in paragraphs:
            if paragraph.strip():
                first_paragraph = paragraph.strip()
                first_paragraph_found = True
                break

        if first_paragraph_found and 'sgpt' in first_paragraph and rerun:
            print(f"Updating CTA:\t {filename}")
            if not dryrun:
                # print("shell gpt paragraph found. updating it.")
                # Remove the first paragraph (up to the first double line break)
                file_content = Path(filename).read_text()
                replace = build_paragraph(file_content) 
                replace = "<!--sgpt-->"+shorter(replace)
                rest_of_article = rest_of_file.lstrip().split("\n\n", 1)[1]
                new_content = frontmatter + '\n' + replace + '\n\n' + rest_of_article
                with open(filename, 'w') as file:
                    file.write(new_content)
        elif 'https://earthly.dev/' not in first_paragraph and 'earthly.dev' not in first_paragraph:
            print(f"Adding CTA:\t {filename}")
            if not dryrun:
                replace = build_paragraph(filename) 
                replace = "<!--sgpt-->"+shorter(replace)
                new_content = frontmatter + '\n' + replace + '\n\n' + rest_of_file.strip()
                with open(filename, 'w') as file:
                    file.write(new_content)

def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

examples = [
    {'input': dedent("""
    **We're [Earthly](https://earthly.dev/). We make building software simpler and therefore faster using containerization. This article is about 5 Blogs for Scala's Birthday. Earthly is a powerful build tool that can be used in conjunction with Scala projects, making it a valuable tool for developers interested in building and managing their Scala code efficiently. [Check us out](/).**
    """),
    'output': """
    **We're [Earthly](https://earthly.dev/). We make building software simpler and therefore faster using containerization. Earthly is a powerful build tool that can be used with Scala projects. [Check it out](/).**
    """},

    {'input': dedent("""
    **We're [Earthly](https://earthly.dev/). We make building software simpler and therefore faster using containerization. This article is about installing `matplotlib` in a Docker container. Earthly is a powerful build tool that can greatly simplify the process of building and managing Docker containers, making it an ideal tool for readers interested in installing `matplotlib` in a Docker container. [Check us out](/).**
    """),
    'output': """
    **We're [Earthly](https://earthly.dev/). We make building software simpler and therefore faster using containerization. It's an ideal tool for dealing with your python container builds. [Check it out](/).**
"""},
]

def shorter(input: str) -> str:
    score = guidance(dedent('''
    {{#system~}}
    Background:
    ---
    Here are some key things to know about Earthly and why it is used in tech:

    Earthly is an open source build automation tool released in 2020. It allows you to define builds using a domain-specific language called Earthfile.

    Key features of Earthly include:

    Reproducible builds - Earthly containers isolate dependencies and build steps so you get consistent, repeatable builds regardless of the environment.
    Portable builds - Earthfiles define each step so you can build on any platform that supports containers like Docker.
    Declarative syntax - Earthfiles use a simple, declarative syntax to define steps, reducing boilerplate.
    Built-in caching - Earthly caches steps and layers to optimize incremental builds.
    Parallel builds - Earthly can distribute work across containers to build parts of a project in parallel.
    Reasons developers use Earthly:

    Simpler configuration than bash scripts or Makefiles.
    Avoid dependency conflicts by isolating dependencies in containers.
    Consistent builds across different environments (local dev, CI, production).
    Efficient caching for faster build times.
    Can build and integrate with any language or framework that runs in containers.
    Integrates with CI systems like GitHub Actions.
    Enables building complex multi-language, multi-component projects.
    Earthly is commonly used for building, testing and shipping applications, especially those with components in different languages. It simplifies build automation for monorepos and complex projects.

    Overall, Earthly is a developer-focused build tool that leverages containers to provide reproducible, portable and parallel builds for modern applications. Its declarative Earthfile syntax and built-in caching help optimize build performance.
    Earthly helps with continuous development but not with continuous deployment and works with any programming language.  
    Earthly helps with build software on linux, using containers. It doesn't help with every SDLC process, but it improves build times which can help other steps indirectly.
    ---

    Task:
    Write a shortened version of this call to action for getting readers of an article that is about another topic interested in Earthly. 
    A great call to action explains why Earthly might interest them by connecting to the topic of the article. But if the connection is not clear, a straight-forward request to look at Earthly is second best.
    Shorter and casual is preffered and the benefits of Earthly or Description of Earthly can be changed to better match the topic at hand.
    {{~/system}}
    {{~#each examples}}
    {{#user~}}
    {{this.input}}
    {{~/user}}
    {{#assistant~}}
    {{this.output}}
    {{~/assistant}}    
    {{~/each}}
    {{#user~}} 
    {{input}}
    {{~/user}}
    {{#assistant~}}
    {{gen 'options' n=7 temperature=0.9 max_tokens=500}}
    {{~/assistant}}
    {{#user~}}
    Can you please comment on the pros and cons of each of these replacements?

    Shorter is better. More connected to the topic at hand is better. Natural sounding, like a casual recommendation is better.
    Overstating things, with many adjectives, is worse. Implying Earthly does something it does not is worse. 
    ---{{#each options}}
    Option {{@index}}: {{this}}{{/each}}
    ---
    {{~/user}}
    {{#assistant~}}
    {{gen 'thinking' temperature=0 max_tokens=2000}}
    {{~/assistant}}
    {{#user~}} 
    Please return the text of the best option, based on above thinking.
    {{~/user}}
    {{#assistant~}}
    {{gen 'answer' temperature=0 max_tokens=500}}
    {{~/assistant}}
    '''), llm=gpt4, silent=True)
    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
        out = score(examples=examples,input=input) 
    return out["answer"].strip()


    

def main():
    parser = argparse.ArgumentParser(description='Add an excerpt to a markdown file.')
    parser.add_argument('--dir', help='The directory containing the markdown files.')
    parser.add_argument('--file', help='The path to a single markdown file.')
    parser.add_argument('--dryrun', help='Dry run mode', action='store_true')

    args = parser.parse_args()

    if args.dryrun:
        print("Dryrun mode activated. No changes will be made.")

    if args.dir:
        for root, dirs, files in os.walk(args.dir):
            for file in files:
                if file.endswith('.md'):
                    path = os.path.join(root, file)
                    add_paragraph_if_word_missing(os.path.join(root, file), args.dryrun)
    elif args.file:
        add_paragraph_if_word_missing(args.file, args.dryrun)
    else:
        print("Please provide either --dir or --file.")
        exit(1)

if __name__ == "__main__":
    main()
