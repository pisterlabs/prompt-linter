import openai
import requests
from bs4 import BeautifulSoup

# Replace with your OpenAI API key
openai.api_key = "YOUR_API_KEY"

def query_arxiv_api(search_query, max_results=5):
    base_url = 'http://export.arxiv.org/api/query?'
    query = f'search_query=all:{search_query}&start=0&max_results={max_results}'
    url = base_url + query
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'xml')
        entries = soup.find_all('entry')

        articles = []
        for entry in entries:
            title = entry.title.text.strip()
            url = entry.id.text.strip()
            published = entry.published.text.strip()

            articles.append({
                'title': title,
                'url': url,
                'published': published
            })

        return articles
    else:
        return None

def generate_gpt3_response(prompt, model="text-davinci-002", temperature=0.7):
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response.choices[0].text.strip()

def fetch_articles_and_generate_response(search_query):
    articles = query_arxiv_api(search_query)
    if articles:
        prompt = f"Please provide an answer to the question '{search_query}' using the following {len(articles)} articles as references, and include a citation for each relevant article, and explain the reasoing behind your response:\n\n"
        for article in articles:
            prompt += f"Title: {article['title']}\nURL: {article['url']}\nPublished: {article['published']}\n\n"

        response = generate_gpt3_response(prompt)
        return response
    else:
        return f"Sorry, I couldn't find any articles related to '{search_query}'."

def chat_with_bot():
    print("Welcome to the ArXiv Chatbot! Ask me complex questions about scientific papers in the database.")
    while True:
        user_input = input("\nAsk your question or type 'exit' to stop: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        else:
            response = fetch_articles_and_generate_response(user_input)
            print(response)

if __name__ == "__main__":
    chat_with_bot()
