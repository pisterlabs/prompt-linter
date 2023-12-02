from openai_afc import AutoFnChatCompletion, AutoFnDefinition, AutoFnParam

def search_google(query: str):
    print('Searching for:', query)
    return ['https://foodexperts.net/bestfruitsofalltime', 'https://youtube.com']

def scrape_webpage(url: str):
    print('Scraping webpage:', url)
    return 'mango'

funcs = [
    AutoFnDefinition(search_google, description='make a google search query', params=[
        AutoFnParam('query', {'type': 'string'})
    ]),
    AutoFnDefinition(scrape_webpage, description='scrape content of a webpage by url', params=[
        AutoFnParam('url', {'type': 'string'})
    ])
]

completion = AutoFnChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Search, what is the most delicious fruit?"}
    ],
    functions=funcs
)

print(completion['choices'][0]['message']['content'])