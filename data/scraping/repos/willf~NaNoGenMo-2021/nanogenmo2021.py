import json
import sys
import os
import random
import argparse
import re
import urllib.parse

from dotenv import load_dotenv

load_dotenv()
import openai


def split_line(line):
    """
    Given a line like | ''[[813 (film)|813]]'' || [[Charles Christie]], [[Scott Sidney]] || [[Wedgwood Nowell]], [[Ralph Lewis (actor)|Ralph Lewis]], [[Wallace Beery]], [[Laura La Plante]] || Mystery || [[Film Booking Offices of America|FBO]]
    Retrun a list of the strings between '||'
    """
    return [cell.strip() for cell in line.split("||")]


def resolve_wikipedia_link(link):
    """
    Given a link like [[813 (film)|813]]
    Return the string 813
    """
    link = link.strip()
    # first remove the brackets
    if link.startswith("[[") and link.endswith("]]"):
        link = link.strip()[2:-2]
    # split the link into the title and the link if there is one
    _, link = link.split("|") if "|" in link else (None, link)
    return link.strip()


def convert_wiki_link_to_md(link):
    """
    Given a link like [[813 (film)|813]]
    Return the string [813](https://en.wikipedia.org/wiki/813 (film))
    """
    link = link.strip()
    # first remove the brackets
    if link.startswith("[[") and link.endswith("]]"):
        link = link.strip()[2:-2]
    # split the link into the title and the link if there is one
    link, title = link.split("|") if "|" in link else (None, link)
    if not link:
        link = title
    link = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(link.strip())}"
    return f"[{title}]({link})"


def group_2(matchobj):
    return matchobj.group(2)


def remove_wikipedia_links(text):
    """
    Given a text, remove wikipedia links
    """
    pat = re.compile(r"\[\[(?:([^\|\]]+)\|)?([^\]]+)\]\]")
    return re.sub(pat, group_2, text)


def resolve_wikipedia_links(cell_text):
    """
    Given the cell text, resolve the wikipedia links
    """
    cells = [cell.strip() for cell in cell_text.split(",")]
    results = [resolve_wikipedia_link(cell) for cell in cells]
    return results


def create_movie_dict(line):
    """
    Given a line like | ''[[813 (film)|813]]'' || [[Charles Christie]], [[Scott Sidney]] || [[Wedgwood Nowell]], [[Ralph Lewis (actor)|Ralph Lewis]], [[Wallace Beery]], [[Laura La Plante]] || Mystery || [[Film Booking Offices of America|FBO]]
    Return a dictionary of the movie
    """
    cells = split_line(line)
    movie_dict = {
        "wiki_link": cells[0],
        "title": remove_wikipedia_links(cells[0]),
        "directors": resolve_wikipedia_links(cells[1]),
        "actors": resolve_wikipedia_links(cells[2]),
        "genre": remove_wikipedia_links(cells[3]),
    }
    return movie_dict


def create_and_list(list):
    """
    Given a list of 1 items, return item
    Given a list of 2 items, return item[0] and item[1]
    Given a list of n items, return item[0], item[1], ...,  and item[-1]
    """
    if len(list) == 1:
        return list[0]
    elif len(list) == 2:
        return list[0] + " and " + list[1]
    else:
        comma_separated = ", ".join(list[:-1])
        return f"{comma_separated}, and {list[-1]}"


def create_prompt(movie_dict):
    """
    Given a movie dictionary, create a prompt
    """
    title = movie_dict["title"]
    directors = create_and_list(movie_dict["directors"])
    actors = create_and_list(movie_dict["actors"])
    genre = movie_dict["genre"].lower()
    prompt = f"{title} is a {genre} movie directed by {directors}. It stars {actors}. Give a synopsis of the movie."
    return prompt


def create_header(movie_dict):
    """
    Given a movie dictionary, create a prompt
    """
    title = movie_dict["title"]
    header = f"## {title}"
    return header


def create_brief_summary(movie_dict):
    """
    Given a movie dictionary, create a brief summary
    """
    title = movie_dict["title"]
    directors = create_and_list(movie_dict["directors"])
    actors = create_and_list(movie_dict["actors"])
    genre = movie_dict["genre"].lower()
    an = "an" if genre[0] in "aeiou" else "a"
    if genre == "adventure":
        genre = "adventure film"
    elif genre == "horror":
        genre = "horror film"
    summary = f"*{title}* is {an} {genre} directed by {directors}. It stars {actors}."
    return summary


def create_title_json_file(input_file, output_file):
    """
    Given an input file, create a json file
    """
    with open(input_file, "r") as f:
        lines = f.readlines()
        dicts = [create_movie_dict(line) for line in lines]
        with open(output_file, "w") as g:
            for dict in dicts:
                json.dump(dict, g)
                g.write("\n")


def create_wiki_link(dict_file):
    wiki_link = dict_file.get("wiki_link")
    if wiki_link:
        link = convert_wiki_link_to_md(wiki_link)
        return f"Wikipedia: {link}"
    return ""


def create_story(dict_file):
    header = create_header(dict_file)
    summary = create_brief_summary(dict_file)
    synopsis = dict_file.get("synopsis")
    if synopsis is None:
        synopsis = "No synopsis available."
    return (
        header
        + "\n\n"
        + summary
        + "\n\n**Synopsis**: "
        + synopsis.strip()
        + "\n\n"
        + create_wiki_link(dict_file)
    )


def create_stories(dict_file, story_file, n):
    with open(dict_file, "r") as f:
        dicts = [json.loads(line) for line in f.readlines()]
        if n == 0:
            n = len(dicts)
        # select n random movies
        dicts = [dicts[i] for i in sorted(random.sample(range(len(dicts)), n))]
        # create a story
        with open(story_file, "w") as g:
            g.write("# Possible Movies\n\n")
            for dict in dicts:
                g.write(create_story(dict))
                g.write("\n\n")


def add_synopsis(movie_dict):
    """
    Given a movie dictionary, create a synopsis from OpenAI
    """
    if "synopsis" in movie_dict:
        return movie_dict
    prompt = create_prompt(movie_dict)
    synopsis = None
    openai.api_key = os.getenv("OPENAI_API_KEY")
    try:
        response = openai.Completion.create(
            engine="davinci-instruct-beta", prompt=prompt, max_tokens=350, n=1
        )
        synopsis = response.choices[0].text
    except openai.exceptions.OpenAIException as err:
        print(err)
    if synopsis is not None:
        movie_dict["synopsis"] = synopsis
    return movie_dict


def add_wikipedia_link(line, movie_dict):
    """
    Given a movie dictionary, add wikipedia links
    """
    cells = split_line(line)
    wiki_link = cells[0]
    movie_dict["wiki_link"] = wiki_link
    return movie_dict


def create_wikipedia_json_file(lines_file, json_file, output_file):
    """
    Given an input file, create a json file
    """
    lines = open(lines_file, "r").readlines()
    dicts = [json.loads(line) for line in open(json_file, "r").readlines()]

    with open(output_file, "w") as g:
        for line, dict in zip(lines, dicts):
            dict = add_wikipedia_link(line, dict)
            json.dump(dict, g)
            g.write("\n")


def create_synopsis_json_file(input_file, output_file, n=0):
    """
    Given an input file, create a json file with synopses
    """
    print("Creating synopses")
    with open(input_file, "r") as f:
        dicts = [json.loads(line) for line in f.readlines()]
        if n == 0:
            n = len(dicts)
        print(f"creating {n} stories from {len(dicts)} entries.")
        j = 0
        with open(output_file, "w") as g:
            for dict in dicts:
                j += 1
                if j <= n:
                    print(f"Getting synopsis {j} of {n}...")
                    dict = add_synopsis(dict)
                    print(dict.get("synopsis"))
                json.dump(dict, g)
                g.write("\n")


import unittest


class TestNanoGenMo2021(unittest.TestCase):
    def test_split(self):
        """
        UnitTest:
        """
        line = """[[813 (film)|813]] || [[Charles Christie]], [[Scott Sidney]] || [[Wedgwood Nowell]], [[Ralph Lewis (actor)|Ralph Lewis]], [[Wallace Beery]], [[Laura La Plante]] || Mystery || [[Film Booking Offices of America|FBO]]"""
        split = [
            "[[813 (film)|813]]",
            "[[Charles Christie]], [[Scott Sidney]]",
            "[[Wedgwood Nowell]], [[Ralph Lewis (actor)|Ralph Lewis]], [[Wallace Beery]], [[Laura La Plante]]",
            "Mystery",
            "[[Film Booking Offices of America|FBO]]",
        ]
        self.assertEqual(split_line(line), split)

    def test_resolve_link(self):
        """
        UnitTest:
        """
        link = "[[813 (film)|813]]"
        self.assertEqual(resolve_wikipedia_link(link), "813")

    def test_resolve_link_simple(self):
        """
        UnitTest:
        """
        link = "[[   813]]"
        self.assertEqual(resolve_wikipedia_link(link), "813")

    def test_resolve_links(self):
        """
        UnitTest:
        """
        cell_text = "[[813 (film)|813]], [[Charles Christie]], [[Scott Sidney]]"
        self.assertEqual(
            resolve_wikipedia_links(cell_text),
            ["813", "Charles Christie", "Scott Sidney"],
        )

    def test_remove_links(self):
        """
        UnitTest:
        """
        cell_text = "Comedy [[short (film)|short]]"
        self.assertEqual(remove_wikipedia_links(cell_text), "Comedy short")

    def test_create_movie_dict(self):
        """
        UnitTest:
        """
        line = """[[813 (film)|813]] || [[Charles Christie]], [[Scott Sidney]] || [[Wedgwood Nowell]], [[Ralph Lewis (actor)|Ralph Lewis]], [[Wallace Beery]], [[Laura La Plante]] || Mystery || [[Film Booking Offices of America|FBO]]"""
        movie_dict = {
            "title": "813",
            "directors": ["Charles Christie", "Scott Sidney"],
            "actors": [
                "Wedgwood Nowell",
                "Ralph Lewis",
                "Wallace Beery",
                "Laura La Plante",
            ],
            "genre": "Mystery",
        }

        self.assertEqual(create_movie_dict(line), movie_dict)

    def test_create_and_list(self):
        """
        UnitTest:
        """
        self.assertEqual(create_and_list(["a"]), "a")
        self.assertEqual(create_and_list(["a", "b"]), "a and b")
        self.assertEqual(create_and_list(["a", "b", "c"]), "a, b, and c")
        self.assertEqual(create_and_list(["a", "b", "c", "d"]), "a, b, c, and d")

    def test_create_prompt(self):
        """
        UnitTest:
        """
        movie_dict = {
            "title": "813",
            "directors": ["Charles Christie", "Scott Sidney"],
            "actors": [
                "Wedgwood Nowell",
                "Ralph Lewis",
                "Wallace Beery",
                "Laura La Plante",
            ],
            "genre": "Mystery",
        }
        prompt = "813 is a mystery movie directed by Charles Christie and Scott Sidney. It stars Wedgwood Nowell, Ralph Lewis, Wallace Beery, and Laura La Plante. Give a synopsis of the movie."
        self.assertEqual(create_prompt(movie_dict), prompt)


def main():
    """
    Main function
    """
    # create_wikipedia_json_file("table.txt", "synopsis.json", "new.json")
    # sys.exit(0)
    # Instantiate the parser
    main_parser = argparse.ArgumentParser(description="@willf nanogenmo2021")
    main_parser.add_argument(
        "--table", type=str, default="table.txt", help="input wiki file"
    )
    main_parser.add_argument(
        "--json", type=str, default="table.json", help="JSON file pre-synopses"
    )
    main_parser.add_argument(
        "--synopsis", type=str, default="synopsis.json", help="JSON file with Synopsis"
    )
    main_parser.add_argument(
        "--story", type=str, default="nanogenmo2021.md", help="story file"
    )
    main_parser.add_argument(
        "--n", type=int, default=1, help="number of stories to generate"
    )

    main_parser.add_argument(
        "--create_json", action="store_true", help="create json file"
    )
    main_parser.add_argument(
        "--create_story", action="store_true", help="create story file"
    )
    main_parser.add_argument(
        "--create_synopses", action="store_true", help="create synopsis file"
    )
    options = main_parser.parse_args()

    if options.table and options.json and options.create_json:
        create_title_json_file(options.table, options.json)
        sys.exit(0)
    if options.story and options.json and options.create_story:
        create_stories(options.synopsis, options.story, options.n)
        sys.exit(0)
    if options.synopsis and options.json and options.create_synopses:
        create_synopsis_json_file(options.json, options.synopsis, options.n)
        sys.exit(0)
    print(
        "no action specified (one of --test, --create_prompts, --create_story, --create_synopses)"
    )
    main_parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
