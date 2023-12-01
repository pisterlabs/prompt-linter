import openai, re


def add_strong_tags(html_code):
    # Use regex to find all h2, h3, and h4 tags
    regex_pattern = r'<(h1|h2|h3|h4)>(.*?)</\1>'
    matches = re.findall(regex_pattern, html_code, flags=re.IGNORECASE)
    
    # Loop through matches and replace the tags with tags wrapped in strong tags
    for match in matches:
        replacement = f'<strong><{match[0]}>{match[1]}</{match[0]}></strong>'
        html_code = html_code.replace(f'<{match[0]}>{match[1]}</{match[0]}>', replacement)
    
    return html_code


def generate_article(topic,  openai_api_key,points, max_tokens):
    # Connect to the OpenAI API
    openai.api_key = openai_api_key
    # print(openai.api_key)
    # Set the desired parameters for the ChatGPT API call
    prompt = f"Please write an SEO-friendly article on {topic}. Article should have introduction,\
    {points}, conclusion, and FAQ. These points should be wrapped in an h2 or h3 HTML tag as necessary.\
    Every header tag should be strong and bold."

    # Make the API call to ChatGPT
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=0.7,
    )

    # Retrieve the generated article from the API response
    article = response.choices[0].text.strip()

    return article
    processed_article = add_strong_tags(article)

    return processed_article
