import re
from urllib.parse import urlparse
from openaikey import openai


def urls(request):
    prompt = f"""give me the urls needed for a chrome: {request}"""
    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[
            {"role": "user", "content": prompt},
        ],
        top_p=0.001,
    )
    # print(response["choices"][0]["message"]["content"])
    return response["choices"][0]["message"]["content"]


def clean_message(request):
    pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    http_links = re.findall(pattern, urls(request))
    return http_links


def url_corr(urls_lst):
    parsed_urls = [urlparse(url) for url in urls_lst]
    netloc_set = set([parsed_url.netloc for parsed_url in parsed_urls])

    if len(netloc_set) > 1:
        return 1
    elif urls_lst:
        return urls_lst
    else:
        return 0


# command = process_input
# print(command)

# # print(clean_message("buy something on amazon"))
# print(url_corr(clean_message("open gmail")))
