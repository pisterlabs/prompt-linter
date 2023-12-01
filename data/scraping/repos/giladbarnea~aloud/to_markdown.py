import builtins
import re
import textwrap
from concurrent.futures import ThreadPoolExecutor

import html2text
from openai import OpenAI
from rich import get_console

console = get_console()
module_cache = {}


def convert_to_raw_markdown(html: str, *, ignore_links: bool = True) -> str:
    md_converter = html2text.HTML2Text(bodywidth=0)
    md_converter.ignore_links = ignore_links
    markdown = md_converter.handle(html).strip()
    markdown = re.sub(r"\n\n\n+", "\n\n", markdown)
    markdown = re.sub(r"  +", " ", markdown)
    markdown = "\n".join(line.strip() for line in markdown.splitlines())
    return markdown


def to_markdown(html: str, *, ignore_links: bool = True) -> str:
    markdown = convert_to_raw_markdown(html, ignore_links=ignore_links)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first_real_article_line_future = executor.submit(get_first_real_article_line, markdown)
        last_real_article_line_future = executor.submit(get_last_real_article_line, markdown)
    first_real_article_line = first_real_article_line_future.result()
    last_real_article_line = last_real_article_line_future.result()
    clean_markdown = remove_website_top_junk(markdown, first_real_article_line)
    clean_markdown = remove_website_bottom_junk(clean_markdown, last_real_article_line)
    return clean_markdown.strip()


def get_first_real_article_line(markdown: str) -> str:
    prompt = textwrap.dedent("""
    You are given a markdown representation of an article from the internet, generated automatically by a tool. This means that the markdown is not perfect.
    Often, the markdown will start with things that used to be the website's navigation bar, social media links, etc, and only after that will the actual article start, usually with a title.
    Find the line where the real article starts, and return exactly that line, and only it, without explanation or anything else.

    The article's markdown representation is:
    ```md
    {markdown}
    ```
    """).format(markdown=markdown).strip()
    if not (oai := module_cache.get("oai")):
        oai = OpenAI()
        module_cache["oai"] = oai
    chat_completion = oai.chat.completions.create(
        messages=[{"role": "system", "content": prompt}],
        model="gpt-4-1106-preview",
        temperature=0,
        stream=False,
        timeout=10,
    )
    first_real_article_line = chat_completion.choices[0].message.content.splitlines()[0]
    return first_real_article_line


def get_last_real_article_line(markdown: str) -> str:
    prompt = textwrap.dedent("""
    You are given a markdown representation of an article from the internet, generated automatically by a tool. This means that the markdown is not perfect.
    Often, at the bottom of the article, the article's real actual content would end, and after that, things that used to be the website's comment section, social media links, navigation elements and buttons would appear. Those elements are called "junk elements".
    Find the last line of the real content, just before where the "junk elements" appear, and return exactly that last real content line, and only it, without explanation.
    If the article does not contain "junk elements", your instruction stays the same: return the last line.

    The article's markdown representation is, enclosed in triple backticks:
    ```md
    {markdown}
    ```
    """).format(markdown=markdown.strip()).strip()
    if not (oai := module_cache.get("oai")):
        oai = OpenAI()
        module_cache["oai"] = oai
    chat_completion = oai.chat.completions.create(
        messages=[{"role": "system", "content": prompt}],
        model="gpt-4-1106-preview",
        temperature=0,
        stream=False,
        timeout=10,
    )
    last_real_article_line = chat_completion.choices[0].message.content.splitlines()[0]
    return last_real_article_line


def remove_website_top_junk(markdown: str, first_real_article_line: str) -> str:
    markdown_lines = markdown.splitlines()
    first_real_article_line_index = index_of(markdown_lines, first_real_article_line)
    console.log(f"first_real_article_line (idx {first_real_article_line_index}):\n{first_real_article_line!r}")
    clean_markdown = "\n".join(markdown_lines[first_real_article_line_index:])
    return clean_markdown


def remove_website_bottom_junk(markdown: str, last_real_article_line: str) -> str:
    markdown_lines = markdown.splitlines()
    reversed_markdown_lines = list(reversed(markdown_lines))
    last_real_article_line_index = index_of(reversed_markdown_lines, last_real_article_line)
    console.log(f"last_real_article_line (idx {last_real_article_line_index}):\n{last_real_article_line!r}")
    if last_real_article_line_index == 0:
        return markdown
    clean_markdown = "\n".join(markdown_lines[:-last_real_article_line_index])
    return clean_markdown


def index_of(string_lines: list[str], substring: str, *, case_sensitive=True) -> int:
    lines_equal_to_substring = [line for line in string_lines if line == substring]
    if lines_equal_to_substring:
        return string_lines.index(lines_equal_to_substring[0])
    lines_starting_with_substring = [line for line in string_lines if line.startswith(substring)]
    if lines_starting_with_substring:
        return string_lines.index(lines_starting_with_substring[0])
    lines_containing_substring = [line for line in string_lines if substring in line]
    if lines_containing_substring:
        return string_lines.index(lines_containing_substring[0])
    if case_sensitive:
        return index_of([line.lower() for line in string_lines], substring.lower(), case_sensitive=False)
    hasattr(builtins, "live") and builtins.live.stop()
    breakpoint()
