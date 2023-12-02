import openai
import config

openai.api_key = config.api_key

class BlogWriter():
    def writeBlog(title='', topic='', emulation='', keywords='', links='', length=''):
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{'role':'user', "content": f"Write a {length} length blog about {topic} with the title {title}. Have it emulate {emulation}. Include the keywords {keywords}. Include the links {links}."}],
            )
        content = completion.choices[0].message.content

        return content
