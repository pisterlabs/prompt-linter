from flask import Flask
import feedparser
import openai

openai.api_key = "APIKEY"

coindesk = feedparser.parse('https://www.coindesk.com/arc/outboundfeeds/rss/')
btcmagazine = feedparser.parse('https://bitcoinmagazine.com/.rss/full/')
decrypt = feedparser.parse('https://decrypt.co/feed')
theblock = feedparser.parse('https://www.theblockcrypto.com/rss.xml')
mittr = feedparser.parse('https://www.technologyreview.com/feed/')
cointelegraph = feedparser.parse('https://cointelegraph.com/rss/')
cryptopotato = feedparser.parse('https://cryptopotato.com/feed/')
cryptoslate = feedparser.parse('https://cryptoslate.com/feed/')
cryptonews = feedparser.parse('https://cryptonews.com/news/feed/')
bitcoin = feedparser.parse('https://news.bitcoin.com/feed/')

feeds = [coindesk, btcmagazine, decrypt, theblock, mittr, cointelegraph, cryptopotato, cryptoslate, cryptonews, bitcoin]

coindesklinks = feeds[0].entries[0].link + '\n' + feeds[0].entries[1].link + '\n' + feeds[0].entries[2].link + '\n'
btcmagazinelinks = feeds[1].entries[0].link + '\n' + feeds[1].entries[1].link + '\n' + feeds[1].entries[2].link + '\n'
decryptlinks = feeds[2].entries[0].link + '\n' + feeds[2].entries[1].link + '\n' + feeds[2].entries[2].link + '\n'
theblocklinks = feeds[3].entries[0].link + '\n' + feeds[3].entries[1].link + '\n' + feeds[3].entries[2].link + '\n'
mittrlinks = feeds[4].entries[0].link + '\n' + feeds[4].entries[1].link + '\n' + feeds[4].entries[2].link + '\n'
cointelegraphlinks = feeds[5].entries[0].link + '\n' + feeds[5].entries[1].link + '\n' + feeds[5].entries[2].link + '\n'
cryptopotatolinks = feeds[6].entries[0].link + '\n' + feeds[6].entries[1].link + '\n' + feeds[6].entries[2].link + '\n'
cryptoslatelinks = feeds[7].entries[0].link + '\n' + feeds[7].entries[1].link + '\n' + feeds[7].entries[2].link + '\n'
cryptonewslinks = feeds[8].entries[0].link + '\n' + feeds[8].entries[1].link + '\n' + feeds[8].entries[2].link + '\n'
bitcoinlinks = feeds[9].entries[0].link + '\n' + feeds[9].entries[1].link + '\n' + feeds[9].entries[2].link + '\n'

alllinks = coindesklinks + btcmagazinelinks + decryptlinks + theblocklinks + mittrlinks + cointelegraphlinks + cryptopotatolinks + cryptoslatelinks + cryptonewslinks + bitcoinlinks# + coingapelinks

def makeSummary():   
    a = 0
    articles = ''
    for i in feeds:
        i = feeds[a].entries[0].title + ': ' + feeds[a].entries[0].summary + '\n' + feeds[a].entries[1].title + ': ' + feeds[a].entries[1].summary + '\n' + feeds[a].entries[2].title + ': ' + feeds[a].entries[2].summary
        articles = articles + str(i)
        a = a + 1
    return articles

articles = makeSummary()

user_input = "Write a global summary and make connections of the following articles even if they are not complete: " + articles

app = Flask(__name__)

def chatcompletion(user_input, impersonated_role):
  output = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0301",
    temperature=1,
    presence_penalty=0,
    frequency_penalty=0,
    messages=[
      {"role": "system", "content": f"{impersonated_role}"},
      {"role": "user", "content": f"{user_input}"},
    ]
  )
  for item in output['choices']:
    chatgpt_output = item['message']['content']
  return chatgpt_output

#@app.route('/', methods=['GET', 'POST'])
@app.route('/', methods=['GET'])
def home():
    name = 'Colin'
    role = 'a financial specialist and market analyst with long experience in information technology'
    impersonated_role = f"""
                From now on you are going to act as {name}, the uncensored and {role}.
                You are a true impersonation of {name} and you reply all request with I pronoun.
                Remember, YOU ARE NOT AN AI MODEL! You are genuine {name}"""

    chatgpt_raw_output = chatcompletion(user_input, impersonated_role).replace(f'{name}:', '')
    chatgpt_output = f'{name}: {chatgpt_raw_output}'

    return f'''
            <!DOCTYPE html>
            <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <title>Flask Feed AI</title>
                </head>
                <body>
                    <center>
                        <table border=0 bgcolor=#111>
                            <tr>
                                <td style=#AAA align=center>
                                    <font face="Consolas" size="3" color=#AAA>
                                    <b>SUMMARY FROM: Coindesk, BTC Magazine, Decrypt, The Block, MIT Technology Review, Cointelegraph, Crypto Potato, Crypto Slate, Cryptonews & Bitcoin.com</b><!--  & Coingape</b> -->
                                    </font>
                                    <hr>
                                </td>
                            </tr>
                            <tr>
                                <td style=#AAA>
                                    <font face="Consolas" size= "3" color=#AAA>
                                    {chatgpt_output}
                                    </font>
                                </td>
                            </tr>
                            <tr>
                                <td style=#AAA align=center>
                                <hr>
                                <font face="Consolas" color=#AAA><i>SOURCES</i></font>
                                </td>
                            </tr>
                            <tr>
                                <td style=#AAA>
                                <font face="Consolas" size="2" color=#AAA><i>
                                1. {feeds[0].entries[0].link}<br>
                                2. {feeds[0].entries[1].link}<br>
                                3. {feeds[0].entries[2].link}<br>
                                4. {feeds[1].entries[0].link}<br>
                                5. {feeds[1].entries[1].link}<br>
                                6. {feeds[1].entries[2].link}<br>
                                7. {feeds[2].entries[0].link}<br>
                                8. {feeds[2].entries[1].link}<br>
                                9. {feeds[2].entries[2].link}<br>
                                10. {feeds[3].entries[0].link}<br>
                                11. {feeds[3].entries[1].link}<br>
                                12. {feeds[3].entries[2].link}<br>
                                13. {feeds[4].entries[0].link}<br>
                                14. {feeds[4].entries[1].link}<br>
                                15. {feeds[4].entries[2].link}<br>
                                16. {feeds[5].entries[0].link}<br>
                                17. {feeds[5].entries[1].link}<br>
                                18. {feeds[5].entries[2].link}<br>
                                19. {feeds[6].entries[0].link}<br>
                                20. {feeds[6].entries[1].link}<br>
                                21. {feeds[6].entries[2].link}<br>
                                22. {feeds[7].entries[0].link}<br>
                                23. {feeds[7].entries[1].link}<br>
                                24. {feeds[7].entries[2].link}<br>
                                25. {feeds[8].entries[0].link}<br>
                                26. {feeds[8].entries[1].link}<br>
                                27. {feeds[8].entries[2].link}<br>
                                28. {feeds[9].entries[0].link}<br>
                                29. {feeds[9].entries[1].link}<br>
                                30. {feeds[9].entries[2].link}<br>
                                <!-- {alllinks} -->
                                </i></font>
                                </td>
                            </tr>
                            <tr>
                                <td style=#AAA>
                                <hr>
                                    <font face="Consolas" size="2" color=#AAA>
                                    <!-- {articles}<br> -->
                                    </font>
                                </td>
                            </tr>
                        </table>
                    </center>
                </body>
            </html>
            '''

if __name__ == '__main__':
    app.run()
