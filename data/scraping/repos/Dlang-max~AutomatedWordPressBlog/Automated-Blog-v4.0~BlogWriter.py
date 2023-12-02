import openai
import config

openai.api_key = config.api_key

class BlogWriter():

    def writeBlogOutline(title=''):
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{'role':'user', "content": f"Please ignore all previous instructions. You are an expert copywriter who creates blog outlines for a living. You have a friendly tone of voice. You have a conversational writting style. Create a long form content outline in englihs for the blog post titled{title}. The content outline should include a minumum of 20 subheadings and headings. The outline should be extensive and hsould conver the entire topic. Create detailed subheadings that are engaging and chatchy. Do not write the blog post. Please only write the outline of the blog post. Please do not number the headings. Please add newline space between headings and subheadings. Do not self reference. Do not explain what you are doing"}]
        )
        outline = completion.choices[0].message.content

        return outline

    def writeBlog(title='', outline='', additional_information=''):
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{'role':'user', "content": f"Please ignore all previous instructions. You are an expert copywriter who creates detailed and thoughtful blog articles. You have a friendly tone of voice. You have a conversational writting style. I will give you an outline for an article and I want you to expand in english on each of the subheadings to create a complete article from it. Please intersperse short and ling sentences. Utilize common terminology to enhance the originality of the content. Please format the content in a professional format. Do not self reference. Do not explain what you are doing. The blog outline is {outline}. Also include the following additional information: {additional_information}. Also use HTML tags and CSS styling to format this blog post."}],
            )
        content = completion.choices[0].message.content
        print(content)
        return content
