from bs4 import BeautifulSoup

from urllib.request import urlopen
import cohere

co = cohere.Client("o2KYh1CEVLYwS0ePRO4VmKsIWZuaSuz5cDS1MWjZ")


def get_text(url):
    page = urlopen(url)
    html = page.read().decode("utf-8")
    soup = BeautifulSoup(html, "html.parser")
    main_soup = soup.find('main')
    total = ""
    for text in main_soup.find_all("p"):
        total += text.get_text()
    return total.split(".")


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


def create_summary(l):
    divided = list(divide_chunks(l, 3))
    l_sum = []
    for i in divided:
        i = ".".join(i)
        sum = summarize(i)
        print(sum)
        l_sum.append(sum)
    total_sum = ".".join(l_sum)
    return total_sum


def summarize(string):
    string = string + "\n In summary: \n ...."
    print(string)

    response = co.generate(
        model='xlarge',
        prompt=string,
        max_tokens=40,
        temperature=0.7,
        p=0.85,
        frequency_penalty=0.5,
        stop_sequences=["...."])

    summary = response.generations[0].text
    return summary
