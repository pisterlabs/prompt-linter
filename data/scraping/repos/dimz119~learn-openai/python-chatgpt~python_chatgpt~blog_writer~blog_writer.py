import openai
import os
from jinja2 import Environment, FileSystemLoader

blog_title = "Let's learn Python History"
prompt = f"""
Biography - My name is Joon. I am working as Data engineer and online coding course instructor
Blog Title - {blog_title}
Tags - Python, Technology, Coding, Machine Learning
Write the blog based on the information above, starting with my bio introduction
"""

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  model="text-davinci-003",
  prompt=prompt,
  max_tokens=300,
)
content = response['choices'][0]['text'].replace('\n', '<br />')

# Load the template file
env = Environment(loader=FileSystemLoader('.'))
template = env.get_template('template.html')

# Render the template with the variables
output = template.render(title=blog_title, content=content)

# Write the output to an HTML file
with open('output.html', 'w') as f:
    f.write(output)