import guidance

def is_search(completion):
    return '<search>' in completion

def search(query):
    # Fake search results
    return [{'title': 'How do I cancel a Subscription? | Facebook Help Center',
        'snippet': "To stop a monthly Subscription to a creator: Go to the creator's Facebook Page using the latest version of the Facebook app for iOS, Android or from a computer. Select Go to Supporter Hub. Select . Select Manage Subscription to go to the iTunes or Google Play Store and cancel your subscription. Cancel your Subscription at least 24 hours before ..."},
        {'title': 'News | FACEBOOK Stock Price Today | Analyst Opinions - Insider',
        'snippet': 'Stock | News | FACEBOOK Stock Price Today | Analyst Opinions | Markets Insider Markets Stocks Indices Commodities Cryptocurrencies Currencies ETFs News Facebook Inc (A) Cert Deposito Arg Repr...'},
        {'title': 'Facebook Stock Price Today (NASDAQ: META) Quote, Market Cap, Chart ...',
        'snippet': 'Facebook Stock Price Today (NASDAQ: META) Quote, Market Cap, Chart | WallStreetZen Meta Platforms Inc Stock Add to Watchlist Overview Forecast Earnings Dividend Ownership Statistics $197.81 +2.20 (+1.12%) Updated Mar 20, 2023 Meta Platforms shares are trading... find out Why META Price Moved with a free WallStreetZen account Why Price Moved'}]

search_demo = guidance('''Seach results:
{{~#each results}}
<result>
{{this.title}}
{{this.snippet}}
</result>{{/each}}''')

demo_results = [
    {'title': 'OpenAI - Wikipedia', 'snippet': 'OpenAI systems run on the fifth most powerful supercomputer in the world. [5] [6] [7] The organization was founded in San Francisco in 2015 by Sam Altman, Reid Hoffman, Jessica Livingston, Elon Musk, Ilya Sutskever, Peter Thiel and others, [8] [1] [9] who collectively pledged US$ 1 billion. Musk resigned from the board in 2018 but remained a donor.'},
    {'title': 'About - OpenAI', 'snippet': 'About OpenAI is an AI research and deployment company. Our mission is to ensure that artificial general intelligence benefits all of humanity. Our vision for the future of AGI Our mission is to ensure that artificial general intelligence—AI systems that are generally smarter than humans—benefits all of humanity. Read our plan for AGI'}, 
    {'title': 'Ilya Sutskever | Stanford HAI', 'snippet': '''Ilya Sutskever is Co-founder and Chief Scientist of OpenAI, which aims to build artificial general intelligence that benefits all of humanity. He leads research at OpenAI and is one of the architects behind the GPT models. Prior to OpenAI, Ilya was co-inventor of AlexNet and Sequence to Sequence Learning.'''}
]

s = search_demo(results=demo_results)

practice_round = [
    {'role': 'user', 'content' : 'Who are the founders of OpenAI?'},
    {'role': 'assistant', 'content': '<search>Who are the founders of OpenAI</search>'},
    {'role': 'user', 'content': str(search_demo(results=demo_results))},
    {'role': 'assistant', 'content': 'The founders of OpenAI are Sam Altman, Reid Hoffman, Jessica Livingston, Elon Musk, Ilya Sutskever, Peter Thiel and others.'},
]



chatgpt_searchtool = '''
{{#system~}}
You are a helpful assistant.
{{~/system}}

{{#user~}}
From now on, whenever your response depends on any factual information, please search the web by using the function <search>query</search> before responding. I will then paste web results in, and you can respond.
{{~/user}}

{{#assistant~}}
Ok, I will do that. Let's do a practice round
{{~/assistant}}

{{#each practice}}
{{#if (== this.role "user")}}
{{#user}}{{this.content}}{{/user}}
{{else}}
{{#assistant}}{{this.content}}{{/assistant}}
{{/if}}
{{/each}}

{{#user~}}
That was great, now let's do another one.
{{~/user}}

{{#assistant~}}
Sounds good
{{~/assistant}}

{{#user~}}
{{user_query}}
{{~/user}}

{{#assistant~}}
{{gen "query" stop="</search>"}}{{#if (is_search query)}}</search>{{/if}}
{{~/assistant}}

{{#user~}}
Search results: {{#each (search query)}}
<result>
{{this.title}}
{{this.snippet}}
</result>{{/each}}
{{~/user}}

{{#assistant~}}
{{gen "resolver"}}
{{~/assistant}}
'''

class SearchToolAgentPOC:
    def __init__(self, guidance, tools, num_iter=3):
        self.guidance = guidance
        self.tools = tools
        self.num_iter = num_iter
    
    def __call__(self, query):
        prompt_start = self.guidance(chatgpt_searchtool)
        final_response = prompt_start(
          user_query=query,
          search=search,
          is_search=is_search,
          practice=practice_round)
        history = final_response.__str__()
        return history, final_response['resolver']

chatgpt_test = """
{{#system~}}
You are a full stack developer. You take in JSON data, and generate Markdown. Including utilizing all the features of markdown in a creative way. The markdown features you know: Headings, Styling text, Quoting text, Quoting code, Supported color models, Links, Section links, Relative links, Images, Lists, Task lists, Mentioning people and teams, Referencing issues and pull requests, Referencing external resources, Uploading assets, Using emoji, Paragraphs, Footnotes
{{~/system}}

{{#user~}}
Question: 
Please generate markdown for the following JSON.  
```js
{
  "title" : "How to change the font size in the Visual Studio Code integrated terminal",
  "keywords": ["Visual Studio Code", "VS Code", "Terminal", "Font Size"],
  "steps": [
    "Open Visual Studio Code.",
    "Press `Ctrl + backtick (grave accent)` to open the integrated terminal.",
    "Click on the gear icon located at the top right corner of the terminal panel or right-click inside the terminal.",
    "In the dropdown menu, click on \"Configure Terminal Settings\" or \"Configure Terminal Profile\".",
    "This will open the `settings.json` file with the terminal profile settings.",
    "Look for the \"fontSize\" property in the JSON file.",
    "Modify the \"fontSize\" value to your desired font size. For example, \"fontSize\": 14.",
    "Save the `settings.json` file."
  ]
}
```

{{~/user}}

{{#assistant~}}

Sure here's the markdown:
```
# How to change the font size in the Visual Studio Code integrated terminal

## Keywords
- Visual Studio Code
- VS Code
- Terminal
- Font Size

## Steps 
- Open Visual Studio Code.
- Press Ctrl + backtick (grave accent) to open the integrated terminal.
- Click on the gear icon located at the top right corner of the terminal panel or right-click inside the terminal.
- In the dropdown menu, click on "Configure Terminal Settings" or "Configure Terminal Profile".
- This will open the settings.json file with the terminal profile settings.
- Look for the "fontSize" property in the JSON file.
- Modify the "fontSize" value to your desired font size. For example, "fontSize": 14.
- Save the settings.json file.
```

{{~/assistant}}

{{#user~}}
Great, good job. Now convert the following JSON into markdown:
```js
{{query}}
```js
{{~/user}}

{{#assistant~}}
Sure, here's the JSON converted to markdown using all the features:
```
{{gen 'resolver' temperature=0.83 max_tokens=700 stop='```'}}
```
{{~/assistant}}
"""