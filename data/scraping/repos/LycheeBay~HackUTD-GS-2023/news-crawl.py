from bs4 import BeautifulSoup
import requests
import json
import datetime
from dateutil.relativedelta import relativedelta
import ast
import cohere
import pickle

co = cohere.Client('kIOSMQOoliUQwVGHDCqPTOe5AFOW4lvdkLvgd8ym') # This is your trial API key

def get_cohere(title, text):
    response = co.generate(
        model='command',
        prompt='Please summarize the following noisy but possible news data extracted from web page HTML with title, and extract keywords of the news. The news text can be very noisy due to it is HTML extraction. Give formatted answer such as Summary: ..., Keywords: ... The news is supposed to be for IBM stock. You may put ’N/A’ if the noisy text does not have relevant information to extract.\n Title: ' + title + '\n Text: ' + text,
        max_tokens=300,
        temperature=0.9,
        k=0,
        stop_sequences=[],
        return_likelihoods='NONE')
    print('Prediction: {}'.format(response.generations[0].text))
    return format(response.generations[0].text)
    

ans = []

with open("ibm-google-link.txt", "r") as f:
    for line in f:
        line = line.split(",", maxsplit=2)
        start_date = datetime.datetime.strptime(line[0], "%Y-%m-%d")
        end_date = datetime.datetime.strptime(line[1], "%Y-%m-%d")
        ans += start_date.strftime("%Y-%m-%d") + ", " + end_date.strftime("%Y-%m-%d") + ", "
        month_entry = [start_date, end_date, []]
        for i in range(2, len(line)):
            print(line[2])
            ref = ast.literal_eval(line[2])
            for piece in ref:
                link = piece[1]
                print(piece[0], piece[1])
                if ".pdf" in link:
                    piece.append(None)
                    ans.append(piece)
                else:
                    page = requests.get(link)
                    soup = BeautifulSoup(page.content, 'html.parser')
                    text = soup.body.get_text()
                    piece.append(text.strip().replace("\n", " "))
                    piece.append(get_cohere(piece[0], text))
                    print(text.strip().replace("\n", " "))
                    ans.append(piece)
                # print(soup.prettify())
                # break
            ans += str(ref)

file = open('parsed_result.bin', 'wb')

# dump information to that file
pickle.dump(ans, file)