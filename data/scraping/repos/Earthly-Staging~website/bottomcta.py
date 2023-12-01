import argparse
import subprocess
import os
from typing import Optional
from textwrap import dedent
import contextlib
import yaml
import guidance

gpt4 = guidance.llms.OpenAI("gpt-4")
gpt35turbo = guidance.llms.OpenAI("gpt-3.5-turbo-16k")

should_cache = True
# should_cache = False

def get_summary(lines: str) -> str:
    score = guidance(dedent('''
    {{#system~}}
    You summarize markdown blog posts.
    {{~/system}}
    {{#user~}}
   

    Post:
    ---
    {{lines}} 
    ---
    Can you summarize this blog post?
    I need a two sentence  summary that will make people want to read it that is casual in tone.
    {{~/user}}
    {{#assistant~}}
    {{gen 'answer' temperature=0 max_tokens=100}}
    {{~/assistant}}
    '''), llm=gpt35turbo, caching=should_cache)
    out = run_llm_program(score, lines=lines)
    return out["answer"].strip() 

def add_tie_in(summary: str, conclusion : str) -> str:    
    # print(f"Summary:{summary}")
    print(f"Original Conclusion:{conclusion}")
    betterconclusion = generate_better_conclusion(conclusion)
    print(f"betterconclusion:{betterconclusion}")
    tie_in = generate_tie_in(summary, betterconclusion)
    print(f"Tie In:{tie_in}")
    better_tie_in = generate_better_tie_in(summary, betterconclusion, tie_in)
    print(f"Better Tie In:{better_tie_in}")
    combined = merge_tie_in(summary,betterconclusion, better_tie_in)
    print(f"Combined:{combined}")
    comment = '<!--sgpt-->\n'
    return comment+ combined

def merge_tie_in(summary: str, conclusion : str, tie_in : str) -> str:
    examples = [
        {
        'conclusion': dedent("""
        Awk has more to it than this. There are more built-in variables and built-in functions. It has range patterns and substitution rules, and you can easily use it to modify content, not just add things up.

        If you want to learn more about Awk, [The Awk Programming Language](https://www.amazon.ca/AWK-Programming-Language-Alfred-Aho/dp/020107981X/) is the definitive book. It covers the language in depth. It also covers how to build a small programming language in Awk, how to build a database in Awk, and some other fun projects.
        """),
        'tie_in' : "Also, if you're the type of person who's not afraid to do things on the command line then you might like [Earthly](/)",
        'result' :  dedent("""
        Awk has more to it than this. There are more built-in variables and built-in functions. It has range patterns and substitution rules, and you can easily use it to modify content, not just add things up.

        If you want to learn more about Awk, [The Awk Programming Language](https://www.amazon.ca/AWK-Programming-Language-Alfred-Aho/dp/020107981X/) is the definitive book. It covers the language in depth. It also covers how to build a small programming language in Awk, how to build a database in Awk, and some other fun projects. Also, if you're the type of person who's not afraid to do things on the command line then you might like [Earthly](/)
        """)
        },
         {
        'conclusion': dedent("""
        Awk has more to it than this. There are more built-in variables and built-in functions. It has range patterns and substitution rules, and you can easily use it to modify content, not just add things up.

        If you want to learn more about Awk, [The Awk Programming Language](https://www.amazon.ca/AWK-Programming-Language-Alfred-Aho/dp/020107981X/) is the definitive book. It covers the language in depth. It also covers how to build a small programming language in Awk, how to build a database in Awk, and some other fun projects.
        """),
        'tie_in' : "Also, if you're the type of person who's not afraid to do things on the command line then you might like [Earthly](/)",
        'result' :  dedent("""
        Awk has more to it than this. There are more built-in variables and built-in functions. It has range patterns and substitution rules, and you can easily use it to modify content, not just add things up.

        If you want to learn more about Awk, [The Awk Programming Language](https://www.amazon.ca/AWK-Programming-Language-Alfred-Aho/dp/020107981X/) is the definitive book. It covers the language in depth. It also covers how to build a small programming language in Awk, how to build a database in Awk, and some other fun projects. 
                           
        (Also, if you're the type of person who's not afraid to do things on the command line then you might like [Earthly](/))
        """)
        },
    ]
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
        {{~/system}}
        {{~#each examples}}
        {{#user~}}
        Tie-in:
        {{this.tie_in}}

        Post Conclusion:
        {{this.conclusion}} 
        ---
        Can you add the tie in to the conclusion in a way that makes sense and blends in? Rewrite it if needed. 
        {{~/user}}
        {{#assistant~}}
        {{this.result}}
        {{~/assistant}}    
        {{~/each}}
        {{#user~}}
        Tie-in:
        {{tie_in}}

        Post Conclusion:
        {{conclusion}} 
        ---
        Can you add the tie in to the conclusion in a way that makes sense and blends in? Rewrite it if needed. 
        {{~/user}}
        {{#assistant~}}
        {{gen 'answer' temperature=0 max_tokens=2000}}
        {{~/assistant}}
    '''), llm=gpt4, caching=should_cache)
    out = run_llm_program(score, conclusion=conclusion, tie_in=tie_in, examples=examples, summary=summary)
    return out["answer"].strip()

def generate_tie_in(summary: str, conclusion : str) -> str:
    examples = [
    {'summary': dedent("""
    Learn the basics of Awk, a powerful text processing tool, in this informative article. 
    """),
    'conclusion': dedent("""
    Awk has more to it than this. There are more built-in variables and built-in functions. It has range patterns and substitution rules, and you can easily use it to modify content, not just add things up.

    If you want to learn more about Awk, [The Awk Programming Language](https://www.amazon.ca/AWK-Programming-Language-Alfred-Aho/dp/020107981X/) is the definitive book. It covers the language in depth. It also covers how to build a small programming language in Awk, how to build a database in Awk, and some other fun projects.
    """),
    'result' : "Also, if you're the type of person who's not afraid to do things on the command line then you might like [Earthly](/)"},
 
]
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
        {{~/system}}
        {{~#each examples}}
        {{#user~}}
        Summary:
        {{this.summary}}

        Post Conclusion:
        {{this.conclusion}} 
        ---
        Can write a short sentence with a markdown link to Earthly that will explain to the reader of this article why they might be interested in Earthly?
        {{~/user}}
        {{#assistant~}}
        {{this.result}}
        {{~/assistant}}    
        {{~/each}}
        {{#user~}}
        Summary:
        {{summary}}

        Post Conclusion:
        {{conclusion}} 
        ---
        Can write a short sentence with a markdown link to Earthly that will explain to the reader of this article why they might be interested in Earthly?
        {{~/user}}
        {{#assistant~}}
        {{gen 'answer' temperature=0 max_tokens=100}}
        {{~/assistant}}
    '''), llm=gpt4, caching=should_cache)
    out = run_llm_program(score, conclusion=conclusion, examples=examples, summary=summary) 
    return out["answer"].strip()

def generate_better_tie_in(summary: str, conclusion : str, tie_in : str) -> str:
    examples = [
    {'tie_in': dedent("""
    For more efficient and reproducible build automation in your Linux environment, you might find [Earthly](https://www.earthly.dev/) a valuable tool to explore. 
                      """),
    'better': dedent("""
    Ready to take your Linux build automation to the next level? Check out [Earthly](https://www.earthly.dev/).    
                     """),
    }
]
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
        {{~/system}}
        {{~#each examples}}
        {{#user~}}
        Tie In:
        {{this.tie_in}}
        ---
        Write a shortened version of Tie In for getting readers of an article they've just finsished reading to learn about Earthly. 
        A great call to action explains why Earthly might interest them by connecting to the topic of the article. But if the connection is not clear, a straight-forward request to look at Earthly is second best.
        Shorter and casual is preffered and the benefits of Earthly or Description of Earthly can be changed to better match the topic at hand.
        {{~/user}}
        {{#assistant~}}
        {{this.better}}
        {{~/assistant}}    
        {{~/each}}
        {{#user~}}
        Summary:
        {{summary}}

        Post Conclusion:
        {{conclusion}} 

        Tie In:
        {{tie_in}}
        ---
        Write a shortened version of Tie In for getting readers of an article they've just finsished reading to learn about Earthly. 
        A great call to action explains why Earthly might interest them by connecting to the topic of the article. But if the connection is not clear, a straight-forward request to look at Earthly is second best.
        Shorter and casual is preffered and the benefits of Earthly or Description of Earthly can be changed to better match the topic at hand.
        {{~/user}}
        {{#assistant~}}
        {{gen 'options' n=7 temperature=0.9 max_tokens=100}}
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
        {{gen 'answer' temperature=0 max_tokens=100}}
    {{~/assistant}}
    '''), llm=gpt4, caching=should_cache)
    out = run_llm_program(score, conclusion=conclusion, summary=summary, tie_in=tie_in, examples=examples) 
    return out["answer"].strip()

def generate_better_conclusion(conclusion : str) -> str:
    examples = [
    {'before': dedent("""
        Docker Slim works to optimize your Docker development process, utilizing both static and dynamic analysis to generate information about your Docker resources that can be used to optimize and secure your images. It does this by disposing of miscellaneous packages and files, and streamlining your container to reduce its attack surface and vulnerabilities.

        The advent of containerized applications has helped scale up the development and production process for DevOps teams. However, Docker's containerization is not perfect, and improvements can be made.

        In this article, you learned about Docker Slim and how it can be used to optimize your Docker resources, utilizing the `lint`, `xray`, `profile`, and `build` Docker Slim commands to optimize your Docker images and containers.
                      """),
    'after': dedent("""
        Docker Slim serves as a handy tool to streamline your Docker development process. It acts like a cleaner for your Docker images, eliminating excess and thereby enhancing their efficiency and security. While Docker has revolutionized the world of DevOps, there's always room for refinement. In this article, we've explored how Docker Slim, with its lint, xray, profile, and build commands, can significantly optimize your Docker images and containers.
                     """),
    },{'before': dedent("""
        Blogs are a great way to keep up on what is new and exciting in the Scala community, and I hope this list of some of my favorites is helpful. If you want to get notified about new Scala blog posts, [The Scala Times](http://scalatimes.com/) is a great option. It's how I found many of these articles.
                      """),
    'after': dedent("""
        Blogs are a great way to keep up on what is new and exciting in the Scala community. I'm glad I could share a few of my favorite Scala resources with you all. For regular Scala updates, check out [The Scala Times](http://scalatimes.com/).It's how I found many of these articles. Take care.
                     """),
    }
]
    score = guidance(dedent('''
        {{#system~}}
        You are a friendly AI, helping to write coding tutorials.
        I will give you a tutorial conclusion and you make it shorter, and more to the point. The tone should be informative and not formal.

        A conclusion goes at the end of an article, so should have the perspective of having read the article and wanting a brief summary and next steps.
        {{~/system}}
        {{~#each examples}}
        {{#user~}}
        {{this.before}} 
        {{~/user}}
        {{#assistant~}}
        {{this.after}}
        {{~/assistant}}    
        {{~/each}}
        {{#user~}}
        {{conclusion}} 
        {{~/user}}
        {{#assistant~}}
        {{gen 'options' n=7 temperature=0.9 max_tokens=500}}
        {{~/assistant}}
        {{#user~}}
        Can you please comment on the pros and cons of each of these replacements? 
        How do they work as tutorial conclusions? Are they consise, informative and natural sounding without being too formal or flippant?
        ---{{#each options}}
        Option {{@index}}: {{this}}{{/each}}
        ---
        {{~/user}}
        {{#assistant~}}
        {{gen 'thinking' temperature=0 max_tokens=2000}}
        {{~/assistant}}
        {{#user~}} 
        Please return the text of the best option, based on above thinking.
        Return just the text, not its option number.
        {{~/user}}
        {{#assistant~}}
        {{gen 'answer' temperature=0 max_tokens=500}}
    {{~/assistant}}
    '''), llm=gpt4, caching=should_cache)
    out = run_llm_program(score, conclusion=conclusion, examples=examples) 
    return out["answer"].strip()

def run_llm_program(program, *args, **kwargs):
    with open("log.txt", "a") as f, contextlib.redirect_stdout(
        f
    ), contextlib.redirect_stderr(f):
        return program(*args, **kwargs)

def add_comment_to_section(text_after_last_heading: str, excerpt : str) -> str:
    comment = '<!--sgpt-->\n'
    if comment in text_after_last_heading:
        text_after_last_heading = add_tie_in(excerpt, text_after_last_heading.replace(comment, ''))
        print('Updating...')
    else:
        print('Adding...')
        text_after_last_heading = add_tie_in(excerpt, text_after_last_heading)
    return text_after_last_heading

def skip(fulltext : str, conclusion: str) -> bool:
    if 'Earthly' in conclusion:
        print("Skipping bc Earthly CTA exists")
        return True
    if "funnel:" in fulltext or "News" in fulltext or " Write Outline" in fulltext or "bottomcta: false" in fulltext:
        print("Skipping bc Funnel artile already")
        return True
    return False

def update_text_after_last_heading(filename: str) -> Optional[None]:
    with open(filename, 'r') as f:
        content = f.read()

    excerpt = get_summary(content)
    lines = content.split('\n')
    for i in reversed(range(len(lines))):
        # Check if the line starts with a heading
        if lines[i].startswith('#'):
            # Get all lines after the last heading
            text_after_last_heading = '\n'.join(lines[i+1:])
            # If 'Earthly' is in the text, skip the file
            if skip(content, text_after_last_heading):
                return
            # Separate lines starting with '{%' (includes) and lines starting with '[^' (footnotes)
            include_lines = [line for line in text_after_last_heading.split('\n') if line.strip().startswith('{%')]
            footnote_lines = [line for line in text_after_last_heading.split('\n') if line.strip().startswith('[^')]
            # Remove include lines and footnote lines from the text after the last heading
            text_after_last_heading = '\n'.join(line for line in text_after_last_heading.split('\n') if not (line.strip().startswith('{%') or line.strip().startswith('[^')))

            # Add the comment to the beginning of the section
            text_after_last_heading = add_comment_to_section(text_after_last_heading, excerpt)
            # Add the include lines and footnote lines back
            text_after_last_heading += '\n\n' + '\n'.join(include_lines + footnote_lines)
            lines[i+1:] = text_after_last_heading.split('\n')
            break

    updated_content = '\n'.join(lines)

    with open(filename, 'w') as f:
        f.write(updated_content)



def main():
    parser = argparse.ArgumentParser(description='Add an excerpt to a markdown file.')
    parser.add_argument('--dir', help='The directory containing the markdown files.')
    parser.add_argument('--file', help='The path to a single markdown file.')

    args = parser.parse_args()

    if args.dir:
        # Process each markdown file in the directory
        for root, dirs, files in os.walk(args.dir):
            for file in files:
                if file.endswith('.md'):
                    path = os.path.join(root, file)
                    print(f"Starting: {path}")
                    update_text_after_last_heading(os.path.join(root, file))
                    print(f"Finishing: {path}")
    elif args.file:
        update_text_after_last_heading(args.file)
    else:
        print("Please provide either --dir or --file.")
        exit(1)

if __name__ == '__main__':
    main()


