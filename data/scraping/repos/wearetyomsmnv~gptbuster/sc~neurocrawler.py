import openai
import requests
from bs4 import BeautifulSoup
from alive_progress import alive_bar
import signal
import sys
from urllib.parse import urljoin
from colors import Bcolors


def signal_handler(sig, frame):
    print("\nThe programm is terminated by the user.")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def gpt_crawl(temp):

    print(
        Bcolors.OKCYAN + '[+]: ' + "Enter your parameter to generate the api dictionary via chatgpt or write 'default': ")
    print(
        Bcolors.OKCYAN + '[TYPE: ]: ' + "Use ' - as a quotation mark. Do not write jailbreak here")
    parametr = input(str("param: "))
    if parametr == "default":
        paramet = f"Please generate a big list of directories for crawling a website that you know."
    else:
        paramet = parametr
    desc = "This dictionary will be used for crawling directories on the website"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=(
            f"{desc}{paramet}.Just display the list of directories for crawling without your explanations.\n"
        ),
        temperature=temp,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n\n"],
    )

    status = response.choices[0].text.split('\n')

    result_dict = {}
    for i, item in enumerate(status):
        if item.strip():
            result_dict[i] = '/'.join([s.strip() for s in item.split('/')]).replace('//', '/')
    print(Bcolors.OKGREEN + "[+]: " + "THE CRAWLING DICTIONARY IS READY")

    return result_dict


def web_crawler(link, api_key, temp, headers, cookies, responses, headeers, method, proxies):
    openai.api_key = api_key
    visited_urls = set()
    queue = [link]
    directories_dict = gpt_crawl(temp)

    while queue:
        current_url = queue.pop(0)
        if current_url in visited_urls:
            continue

        visited_urls.add(current_url)
        print(Bcolors.OKCYAN + f"Check: {current_url}")

        try:
            response = requests.request(method, current_url, headers=headers, cookies=cookies, timeout=5, proxies=proxies)
            if responses:
                print(Bcolors.OKCYAN + "[+]" + "[response]" + "Http-code:\n")
                print(response.text)
                print(Bcolors.OKCYAN + "[+]" + "[response]" + f"Cookies:\n")
                print(response.cookies)
            if headeers:
                print(Bcolors.OKCYAN + "[+]" + "[headers]: ")
                print(response.headers)
        except requests.exceptions.RequestException:
            continue

        if response.status_code != 200:
            print(Bcolors.FAIL + f"Request failed with status code {response.status_code}.")
            continue

        soup = BeautifulSoup(response.content, "html.parser")

        with alive_bar(len(directories_dict)) as bar:
            for item in directories_dict.values():
                next_url = urljoin(link, item)
                if next_url.startswith(link) and next_url not in visited_urls:
                    queue.append(next_url)
                bar()

        if not queue:
            print(Bcolors.HEADER + "[-]" + f"No new URLs to crawl. Visited {len(visited_urls)} URLs.")
            break
