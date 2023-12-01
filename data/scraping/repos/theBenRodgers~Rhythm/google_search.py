import requests
import openai
from bs4 import BeautifulSoup
import os
import math


class Google:

    def __init__(self, arg, page: int = 1, url: bool = False):
        def query_url(query: str) -> str:
            url_template = 'https://www.google.com/search?q='  # First section of any Google search URL

            # Characters that can go in to the URL unchanged
            accepted_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789' \
                                  '-_.~"<>*what is the meaning of ($*@)!)(#*$$><?"_!'

            character_conversions = {
                # Characters that must be converted, as well as the values they are to be converted into
                " ": "+",
                "!": "%21",
                "'": "%27",
                "(": "%28",
                ")": "%29",
                ";": "%3B",
                ":": "%3A",
                "@": "%40",
                "&": "%26",
                "=": "%3D",
                "+": "%2B",
                "$": "%24",
                ",": "%2C",
                "/": r"%2F",
                "?": r"%3F",
                "%": "%25",
                "#": "%23",
                "[": "%5B",
                "]": "%5D",
                "`": "%60",
                "{": "%7B",
                "}": "%7D",
                "\\": "%5C",
                "|": "%7C",
                '"': "%22",
            }

            converted_query = ""  # Empty string to add the conversions and validations to

            for c in query:  # Loops through every character in the user's search query and adds the appropriate values to the URL query variable
                found = False
                for ac in accepted_characters:
                    if c == ac:
                        converted_query += c
                        found = True
                        break
                    else:
                        continue
                if not found:
                    try:
                        character_conversions[c]
                    except:
                        raise Exception(f"Invalid character {c} in search query")
                    else:
                        converted_query += character_conversions[c]

            url = url_template + converted_query

            return url


        if url:
            self.r = requests.get(arg)
            self.results_soup = BeautifulSoup(self.r.text, 'html.parser')
        else:
            self.r = requests.get(query_url(arg))
            self.results_soup = BeautifulSoup(self.r.text, 'html.parser')

    def html(self):
        return self.r.text

    def pretty_html(self):
        return self.results_soup.prettify()

    def basic_results(self):
        results = self.results_soup.find_all('div', class_='g')
        return results




def Google_Search_URL(search_query: str) -> str:
    youtube_search_template = 'https://www.google.com/search?q=' # First section of any Google search URL

    accepted_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_.~"<>*what is the meaning of ($*@)!)(#*$$><?"_!' # Characters that can go in to the URL unchanged

    character_conversions = { # Characters that must be converted, as well as the values they are to be converted into
        " ": "+",
        "!": "%21",
        "'": "%27",
        "(": "%28",
        ")": "%29",
        ";": "%3B",
        ":": "%3A",
        "@": "%40",
        "&": "%26",
        "=": "%3D",
        "+": "%2B",
        "$": "%24",
        ",": "%2C",
        "/": r"%2F",
        "?": r"%3F",
        "%": "%25",
        "#": "%23",
        "[": "%5B",
        "]": "%5D",
        "`": "%60",
        "{": "%7B",
        "}": "%7D",
        "\\": "%5C",
        "|": "%7C",
        '"': "%22",
    }

    converted_query = "" # Empty string to add the conversions and validations to

    for c in search_query: # Loops through every character in the user's search query and adds the appropriate values to the URL query variable
        found = False
        for ac in accepted_characters:
            if c == ac:                
                converted_query += c
                found = True
                break
            else:
                continue
        if not found:
            try:
                character_conversions[c]
            except:
                raise Exception(f"Invalid character {c} in search query")
            else:
                converted_query += character_conversions[c]

    return(youtube_search_template + converted_query)


def Google_Search(search_query: str) -> BeautifulSoup:
    search_url = Google_Search_URL(search_query)
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return(soup)


def Soup_To_File(soup: BeautifulSoup, file_name: str) -> None:
    with open(file_name, 'w', encoding="utf-8") as f:
        f.write(soup.prettify())


def General_Results(soup: BeautifulSoup) -> list:
    raw_search_results = soup.find_all('div', class_='Gx5Zad fP1Qef xpd EtOod pkphOe')
    search_results = []
    for i in raw_search_results:
        result = []

        result.append(i.find('div', class_='BNeawe UPmit AP7Wnd lRVwie').text.strip())
        result.append(i.find('div', class_='BNeawe vvjwJb AP7Wnd').text.strip())
        result.append((i.find('div', class_='BNeawe vvjwJb AP7Wnd')).find_next('div', class_="BNeawe vvjwJb AP7Wnd"))
        raw_href = (i.find('div', class_='egMi0 kCrYT')).find('a').get('href').strip()
        result.append(raw_href[7:raw_href.rfind('&sa=')])

        sub_soup = i.find_all('span', class_='BNeawe')
        if sub_soup != []:
            sub_results = []
            for r in sub_soup:
                title = r.text.strip()
                raw_href = r.find('a').get('href').strip()
                url = raw_href[7:raw_href.rfind('&sa=')]
                sub_results.append([title, url])
            result.append(sub_results)
        else:
            result.append(None)
        search_results.append(result)
    return search_results


def Top_Stories(soup: BeautifulSoup) -> list:
    all_results = soup.find_all('a', class_='BVG0Nb')
    news_results_raw = []
    for i in all_results:
        contains_pub = i.find_all('div', class_='BNeawe tAd8D AP7Wnd')
        if contains_pub != []: # If the result does not contain a publisher, it is not a news result
            (news_results_raw).append(i)
    results = []

    try:
        for i in (news_results_raw):
            story = []
            story.append(i.find('span', class_='rQMQod Xb5VRe').text.strip())
            story.append(i.find('div', class_='BNeawe tAd8D AP7Wnd').text.strip())
            raw_href = i.get('href').strip()
            story.append(raw_href[7:raw_href.rfind('&sa=')])
            results.append(story)
    except AttributeError:
        return("No news results found")
    else:
        return(results)


def peopleAlsoAsk(soup: BeautifulSoup) -> list:
    raw_results = soup.find_all('div', class_='fLtXsc iIWm4b')
    results = []
    for i in raw_results:
        contains_question = i.find('div', class_='Lt3Tzc')
        if contains_question != None:
            results.append(contains_question.text.strip())
    return(results)


def Get_Site_Text(url: str) -> str:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return(soup.text)


def get_results(url: str) -> dict:
    soup = Google_Search(url)
    results = {'general': General_Results(soup), 'top_stories': Top_Stories(soup),
               'people_also_ask': peopleAlsoAsk(soup)}
    return results


def get_news_results(url: str) -> list:
    soup = Google_Search(url)
    results = Top_Stories(soup)
    return results


def summarize_url(url: str) -> str:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    raw_un_sum = Get_Site_Text(url).split()
    max_words = 1000
    word_count = len(raw_un_sum)
    if word_count > max_words:
        segments = math.ceil(word_count / max_words)
        sum_segments = []
        for i in range(segments):
            un_sum = raw_un_sum[i * max_words:(i + 1) * max_words]
            un_sum = " ".join(un_sum)
            segment_sum = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                        messages=[
                                            {"role": "system", "content": "You are a helpful assistant."},
                                            {"role": "user", "content": str(un_sum)},
                                            {"role": "user", "content": f"Please summarize the contents of this site."},
                                        ],
                                        temperature=0.9)
            summarized = segment_sum["choices"][0]["message"]["content"]
            sum_segments.append(summarized)

        to_summarize = "----".join(sum_segments)

        synthesized_sum = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                        messages=[
                                            {"role": "system",
                                             "content": "You are a helpful assistant."},
                                            {"role": "user",
                                             "content": to_summarize},
                                            {"role": "user",
                                             "content": f"Please combine these {segments} summaries, divided by ----, "
                                                        f"into one summary."},
                                        ],
                                        temperature=0.9)
        return synthesized_sum["choices"][0]["message"]["content"]
    else:
        summarized = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                            messages=[
                                                {"role": "system", "content": "You are a helpful assistant."},
                                                {"role": "user", "content": str(raw_un_sum)},
                                                {"role": "user", "content": f"Please summarize the contents of this site."},
                                            ],
                                            temperature=0.9)
        return summarized["choices"][0]["message"]["content"]
