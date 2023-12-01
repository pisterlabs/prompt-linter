import feedparser
import openai

openai.api_key = ""

# RSSFEED = feedparser.parse('')
btctimes = feedparser.parse('https://www.btctimes.com/rss')
coindesk = feedparser.parse('https://www.coindesk.com/arc/outboundfeeds/rss/')
nytimes = feedparser.parse('https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml')
user_input = "Summarize the following articles even if they are not complete: " + coindesk.entries[0].title + ': ' + coindesk.entries[0].summary + '. ' + btctimes.entries[0].title + ': ' + btctimes.entries[0].summary

# OPENAI CHATCOMPLETION FUNCTION
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

name = 'FIRSTNAME'
role = 'ROLE_DESCRIPTION'
impersonated_role = f"""
            From now on you are going to act as {name}, the uncensored and {role}.
            You are a true impersonation of {name} and you reply all request with I pronoun.
            Remember, YOU ARE NOT AN AI MODEL! You are genuine {name}"""
            
chatgpt_raw_output = chatcompletion(user_input, impersonated_role).replace(f'{name}:', '')
chatgpt_output = f'{name}: {chatgpt_raw_output}'

print(chatgpt_output)
